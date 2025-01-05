import os
import argparse
import multiprocessing as mp
import numpy as np
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 byte as uint16)
    # construct the tokens numpy array, if not already
    print(f"\nwriting {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def tokenize(doc, tokenizer, max_length):
    return np.array(
        tokenizer.encode(doc["text"], add_special_tokens=True, truncation=True, max_length=max_length),
        dtype=np.uint16
    )


def tokenize_fw(fw, split='train', max_length=2**16):
    # tokenize all documents and write output shards, each of approximately shard_size tokens
    # ensures each shard contains complete sequences only
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        current_shard = []
        current_size = 0
        progress_bar = None
        tokenize_fn = partial(tokenize, tokenizer=tokenizer, max_length=max_length)

        for tokens in pool.imap(tokenize_fn, fw, chunksize=16):
            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

            # If adding this sequence would exceed shard size, write current shard and start new one
            if current_size + len(tokens) > args.shard_size and current_size > 0:
                # Convert accumulated tokens to numpy array and write
                all_tokens_np = np.concatenate(current_shard)
                filename = os.path.join(DATA_CACHE_DIR, f"fwedu_{split}_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens_np)

                # Reset for next shard
                shard_index += 1
                current_shard = []
                current_size = 0
                progress_bar = None

            # Add sequence to current shard
            current_shard.append(tokens)
            current_size += len(tokens)
            if progress_bar:
                progress_bar.update(len(tokens))

        # Write final shard if there are remaining sequences
        if current_size > 0:
            all_tokens_np = np.concatenate(current_shard)
            filename = os.path.join(DATA_CACHE_DIR, f"fwedu_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np)


parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-m", "--max_length", type=int, default=2**16, help="Maximum sequence length")


if __name__ == "__main__":
    args = parser.parse_args()
    local_dir = 'fineweb-edu'

    # create the cache the local directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    train_fw = ds.take(100_000_000)
    valid_fw = ds.take(40_000)
    test_fw = ds.take(40_000)
    tokenize_fw(train_fw, split='train', max_length=args.max_length)
    tokenize_fw(valid_fw, split='valid', max_length=args.max_length)
    tokenize_fw(test_fw, split='test', max_length=args.max_length)
