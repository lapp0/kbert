# Based on projects below:
# https://github.com/KellerJordan/modded-nanogpt
# and https://github.com/Synthyra/SpeedRunningESM2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Tuple
from transformers import AutoTokenizer


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        assert dim % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_attention_heads)
        self.o_proj = CastedLinear(dim, dim)
        self.o_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor, v1: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        q = q.view(B, T, self.num_attention_heads, -1)
        k = k.view(B, T, self.num_attention_heads, -1)
        v = v.view(B, T, self.num_attention_heads, -1)
        if v1 is None:
            v1 = v
        v = self.lambdas[0] * v + self.lambdas[1] * v1.view_as(v)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)  # re-assemble all head outputs side by side
        y = self.o_proj(y)
        return y, v1


class MLP(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.up = CastedLinear(dim, intermediate_dim)
        self.down = CastedLinear(intermediate_dim, dim)
        self.down.weight.data.zero_()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # https://arxiv.org/abs/2109.08668v2
        # ReLU squared ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        return self.down(self.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config.model_dim, config.num_attention_heads)
        self.mlp = MLP(config.model_dim, config.intermediate_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: torch.Tensor, v1: torch.Tensor, x0: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x_out, v1 = self.attn(norm(x), v1, block_mask)
        x += x_out
        x = x + self.mlp(norm(x))
        return x, v1


class KBERT(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.config = config
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_uri)
        self.masker = MLMMasker(tokenizer)
        self.cls_id = tokenizer.cls_token_id
        self.vocab_size = (tokenizer.vocab_size // 256 + 1) * 256  # round up to nearest 256
        self.num_layers = config.num_layers

        # U-net design by @brendanh0gan
        assert config.num_layers % 2 == 0, "Number of layers should be even for U-net design"
        self.num_encoder_layers = config.num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(self.vocab_size, config.model_dim, padding_idx=tokenizer.pad_token_id)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        # U-net structure on token value embeddings by @leloykun
        self.lm_head = CastedLinear(config.model_dim, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        return logits

    def encoder_pass(self, input_ids, sliding_window_size):
        input_ids = input_ids.flatten()
        docs = (input_ids == self.cls_id).cumsum(dim=0)  # shape: [S]

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            return bidirectional_sliding_window_mask & doc_mask

        S = len(input_ids)
        block_mask = create_block_mask(doc_mask_mod, None, None, S, S)

        x = self.embed(input_ids[None]).bfloat16()
        x = norm(x) # @Grad62304977
        x0 = x
        v1 = None  # first layer value residual

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x, v1 = self.blocks[i](x, v1, x0, block_mask)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, _ = self.blocks[self.num_encoder_layers + i](x, v1, x0, block_mask)

        return x

    def forward(
            self,
            input_ids: torch.Tensor,
            sliding_window_size: torch.Tensor,
            mlm_probability: torch.Tensor,
            keep_replace_prob: torch.Tensor) -> torch.Tensor:
        input_ids, labels = self.masker(input_ids, mlm_probability, keep_replace_prob)
        last_hs = self.encoder_pass(input_ids, sliding_window_size)
        logits = self.get_logits(last_hs)
        return self.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())


class MLMMasker(nn.Module):
    def __init__(self, tokenizer):
        """
        Baseline: 80% replaced with [MASK], 10% replaced with a random token, and 10% unchanged.
        """
        super().__init__()
        self.mask_token_id = tokenizer.mask_token_id
        standard_tokens = [tok_id for tok_id in tokenizer.vocab.values() if tok_id not in tokenizer.all_special_ids]
        self.register_buffer("standard_tokens", torch.tensor(standard_tokens, dtype=torch.int32))
        self.register_buffer("special_tokens", torch.tensor(tokenizer.all_special_ids, dtype=torch.int32))

    def __call__(
            self, input_ids: torch.Tensor, mask_prob: torch.Tensor, keep_replace_prob: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()

        # Create special tokens mask using broadcasting
        special_tokens_mask = (input_ids[..., None] == self.special_tokens).any(-1)

        mlm_prob = mask_prob + keep_replace_prob * 2
        mask_portion = mask_prob / mlm_prob

        # Create probability matrix and mask special tokens
        probability_matrix = torch.ones_like(labels, dtype=torch.float) * mlm_prob
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full_like(probability_matrix, mask_portion)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # keep_replace_prob% of the time, we replace masked input tokens with random word
        replacement_idxs = torch.bernoulli(
            torch.full_like(probability_matrix, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_token_idxs = torch.randint(
            0, self.standard_tokens.numel(), (replacement_idxs.sum(),),
            dtype=input_ids.dtype, device=replacement_idxs.device
        )
        input_ids[replacement_idxs] = self.standard_tokens[random_token_idxs]

        # The rest of the time (keep_replace_prob% of the time again) we keep the masked input tokens unchanged
        return input_ids, labels
