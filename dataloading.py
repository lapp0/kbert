from typing import Optional
import torch
from pathlib import Path


def _load_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open('rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, 'number of tokens read does not match header?'
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern: str, local_mini_batch_size: int, rank: int, world_size: int):
        self.world_size = world_size
        self.rank = rank
        self.files = sorted(Path.cwd().glob(filename_pattern))
        self.local_mini_batch_size = local_mini_batch_size  # single accumulation step on a single device
        self.global_mini_batch_size = world_size * local_mini_batch_size  # a single accumulation step across all devices

        self.reset()

    def reset(self):
        self.next_shard = 0
        self.advance()

    def advance(self):
        self.pos = 0
        self.tokens = _load_data_shard(self.files[self.next_shard])
        self.next_shard = (self.next_shard + 1) % len(self.files)

    @torch.no_grad()
    def next_batch(self):
        buf = self.tokens[self.pos + self.rank * self.local_mini_batch_size:][:self.local_mini_batch_size]
        sequence = buf.to(device="cuda", dtype=torch.int32, non_blocking=True)
        # advance current position and load next shard if necessary
        self.pos += self.global_mini_batch_size
        if self.pos + self.global_mini_batch_size >= len(self.tokens):
            self.advance()
        return sequence


class DistributedPaddedDataLoader(DistributedDataLoader):
    def __init__(
            self, filename_pattern, local_mini_batch_size, rank, world_size,
            bos_id, pad_id, max_epochs=1, allow_windowed=False
    ):
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.allow_windowed = allow_windowed
        self.max_epochs = max_epochs
        self._leftover_tokens = torch.empty(0, dtype=torch.uint16)
        self._curr_epoch = 0
        super().__init__(filename_pattern, local_mini_batch_size, rank, world_size)

    def advance(self):
        self.pos = 0

        # handle epoch limit
        if self.max_epochs is None or self.next_shard // len(self.files) < self.max_epochs:
            next_tokens = _load_data_shard(self.files[self.next_shard % len(self.files)])
            raw_tokens = torch.cat([self._leftover_tokens, next_tokens], dim=0)
            self.next_shard += 1
        else:
            next_tokens = None
            raw_tokens = self._leftover_tokens
        if not raw_tokens.numel():
            self._leftover_tokens = torch.empty(0, dtype=torch.uint16)
            self.tokens = torch.empty(0, dtype=torch.uint16)
            return

        curr_epoch = (self.next_shard - 1) // len(self.files)
        if curr_epoch > self._curr_epoch:
            print(f"end of epoch {self._curr_epoch}")
            self._curr_epoch = curr_epoch

        processed_chunks = []
        curr_batch_len = 0

        bos_positions = (raw_tokens == self.bos_id).nonzero(as_tuple=True)[0]
        assert bos_positions[0] == 0

        for i in range(len(bos_positions) - 1):
            if i < len(bos_positions) - 2:
                sample_end = bos_positions[i + 1] - 1
            elif next_tokens is None:
                sample_end = len(raw_tokens)
            else:
                break

            curr_bos = bos_positions[i]
            sample = raw_tokens[curr_bos:sample_end]  # One sample: "CLS ... EOS"

            # if adding sample exceeds the batch size resulting in truncation, pad to end of batch, starting a fresh batch
            if curr_batch_len and len(sample) + curr_batch_len >= self.local_mini_batch_size:
                num_pad = self.local_mini_batch_size - curr_batch_len
                processed_chunks.append(torch.full((num_pad,), self.pad_id, dtype=torch.uint16))
                curr_batch_len = 0

            # if len(sample) > local batch size, chunk evenly, making multiple padded batches, starting a fresh batch
            if len(sample) > self.local_mini_batch_size:
                assert self.allow_windowed, "Sampler larger than local batch size and windowed sampled are illegal"
                for split_sample in torch.chunk(sample, len(sample) // self.local_mini_batch_size + 1):
                    processed_chunks.append(split_sample)
                    num_pad = self.local_mini_batch_size - len(split_sample)
                    processed_chunks.append(torch.full((num_pad,), self.pad_id, dtype=torch.uint16))
                curr_batch_len = 0
                continue

            processed_chunks.append(sample)
            curr_batch_len += len(sample)
            curr_batch_len = 0 if curr_batch_len == self.local_mini_batch_size else curr_batch_len

        if curr_batch_len:
            num_pad = self.local_mini_batch_size - curr_batch_len
            processed_chunks.append(torch.full((num_pad,), self.pad_id, dtype=torch.uint16))

        self._leftover_tokens = torch.empty(0, dtype=torch.uint16) if next_tokens is None else raw_tokens[sample_end + 1:]
        self.tokens = torch.cat(processed_chunks, dim=0)
