# Based these projects, attribution for discovered improvements within
# https://github.com/KellerJordan/modded-nanogpt
# and https://github.com/Synthyra/SpeedRunningESM2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Tuple
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from dataclasses import dataclass


@dataclass
class ModelConfig(PretrainedConfig):
    tokenizer_uri: str = "answerdotai/ModernBERT-base"
    num_layers: int = 12
    num_attention_heads: int = 6
    model_dim: int = 768
    intermediate_dim: int = 768 * 3


@dataclass
class SequenceClassificationModelConfig(ModelConfig):
    num_labels: int


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
        q = self.c_q(x).view(B, T, self.num_attention_heads, -1)
        k = self.c_k(x).view(B, T, self.num_attention_heads, -1)
        v = self.c_v(x).view(B, T, self.num_attention_heads, -1)
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
        x = x + x_out
        x = x + self.mlp(norm(x))
        return x, v1


class KBERTModel(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizer):
        super().__init__(config)

        self.cls_id = tokenizer.cls_token_id
        self.vocab_size = (tokenizer.vocab_size // 256 + 1) * 256  # round up to nearest 256

        # U-net design by with learnable skip connection weights for decoder layers
        self.num_layers = config.num_layers
        assert config.num_layers % 2 == 0, "Number of layers should be even for U-net design"
        self.num_encoder_layers = config.num_layers // 2  # Half of the layers for encoder
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers  # Remaining for decoder
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(self.vocab_size, config.model_dim, padding_idx=tokenizer.pad_token_id)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

    def forward(self, input_ids, sliding_window_size):
        input_ids = input_ids.flatten()
        docs = (input_ids == self.cls_id).cumsum(dim=0)  # shape: [S]

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            return bidirectional_sliding_window_mask & doc_mask

        S = len(input_ids)
        block_mask = create_block_mask(doc_mask_mod, None, None, S, S)

        x = self.embed(input_ids[None]).bfloat16()
        x = norm(x)
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


class KBERTForMaskedLM(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_uri)
        self.masker = MLMMasker(tokenizer)
        self.encoder = KBERTModel(config, tokenizer)
        self.vocab_size = self.encoder.vocab_size
        self.lm_head = CastedLinear(config.model_dim, self.vocab_size)
        self.encoder.embed.weight = self.lm_head.weight  # tie weights

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)
        logits = logits.float()
        return logits

    def forward(
            self,
            input_ids: torch.Tensor,
            sliding_window_size: torch.Tensor,
            mask_prob: torch.Tensor,
            keep_replace_prob: torch.Tensor) -> torch.Tensor:
        input_ids, labels = self.masker(input_ids, mask_prob, keep_replace_prob)
        last_hs = self.encoder(input_ids, sliding_window_size)
        logits = self.get_logits(last_hs)
        return F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())


class KBERTForSequenceClassification(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_uri)
        self.masker = MLMMasker(tokenizer)
        self.model = KBERTModel(config, tokenizer)
        self.classifier = CastedLinear(config.model_dim, config.num_labels)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)
        logits = logits.float()
        return logits

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            sliding_window_size: torch.Tensor) -> torch.Tensor:
        last_hs = self.model(input_ids, sliding_window_size)
        logits = self.get_logits(last_hs)
        return F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())


class MLMMasker(nn.Module):
    def __init__(self, tokenizer):
        """ELECTRA-style MLM objective, replacing mask_prob with mask, and randomly replacing keep_replace_prob"""
        super().__init__()
        self.mask_token_id = tokenizer.mask_token_id
        standard_tokens = [tok_id for tok_id in tokenizer.vocab.values() if tok_id not in tokenizer.all_special_ids]
        self.register_buffer("standard_tokens", torch.tensor(standard_tokens, dtype=torch.int32))
        self.register_buffer("special_tokens", torch.tensor(tokenizer.all_special_ids, dtype=torch.int32))

    def __call__(
            self, input_ids: torch.Tensor, mask_prob: torch.Tensor, keep_replace_prob: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # include mlm_prob tokens in MLM objective
        mlm_prob = mask_prob + 2 * keep_replace_prob
        labels = input_ids.clone()
        special_tokens_mask = (input_ids[..., None] == self.special_tokens).any(dim=-1)
        inclusion_mask = torch.bernoulli((~special_tokens_mask).float() * mlm_prob).bool()
        labels[~inclusion_mask] = -100

        # replace mask_prob tokens with <mask>, keep_replace_prob tokens with random token
        mask_portion = mask_prob / mlm_prob
        replace_with_mask = torch.bernoulli(inclusion_mask.float() * mask_portion).bool()
        replace_with_rand = torch.bernoulli((inclusion_mask & ~replace_with_mask).float() * 0.5).bool()
        random_ids = torch.randint(0, self.standard_tokens.numel(), (replace_with_rand.sum(),), device=labels.device)
        input_ids[replace_with_mask] = self.mask_token_id
        input_ids[replace_with_rand] = self.standard_tokens[random_ids]

        return input_ids, labels
