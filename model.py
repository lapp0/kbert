# Based these projects, attribution for discovered improvements within
# https://github.com/KellerJordan/modded-nanogpt
# and https://github.com/Synthyra/SpeedRunningESM2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Tuple, Optional
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from dataclasses import dataclass, fields


@dataclass
class ModelConfig(PretrainedConfig):
    tokenizer_uri: str = "answerdotai/ModernBERT-base"
    num_layers: int = 20
    num_attention_heads: int = 6
    model_dim: int = 768
    intermediate_dim: int = 2048
    logit_softcap: Optional[int] = 15
    head_dropout: float = 0.0

    def __init__(self, **kwargs):
        # ignore PretrainedConfig implicit attributes
        for f in fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs.pop(f.name))


@dataclass
class SequenceClassificationModelConfig(ModelConfig):
    architectures = ("KBERTForSequenceClassification",)
    num_labels: int


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor):
        cos = self.cos[pos_ids, None, :]
        sin = self.sin[pos_ids, None, :]
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x)


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

    def forward(
            self,
            x: torch.Tensor,
            vr: torch.Tensor,
            block_mask: torch.Tensor,
            pos_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.num_attention_heads, -1)
        k = self.c_k(x).view(B, T, self.num_attention_heads, -1)
        v = self.c_v(x).view(B, T, self.num_attention_heads, -1)
        vr = v if vr is None else vr
        v = self.lambdas[0] * v + self.lambdas[1] * vr.view_as(v)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q, pos_ids), self.rotary(k, pos_ids)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)  # re-assemble all head outputs side by side
        y = self.o_proj(y)
        return y, v


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

    def forward(
            self,
            x: torch.Tensor,
            ve: torch.Tensor,
            x0: torch.Tensor,
            block_mask: torch.Tensor,
            pos_ids: torch.Tensor
    ) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        attn_out, v = self.attn(norm(x), ve, block_mask, pos_ids)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, v


class KBERTModel(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizer):
        super().__init__(config)

        assert config.num_layers % 4 == 0, "num_layers must be divisible by 4 for U-net and value embeddings"

        self.cls_id = tokenizer.cls_token_id
        self.vocab_size = (tokenizer.vocab_size // 256 + 1) * 256  # round up to nearest 256
        self.model_dim = config.model_dim

        self.embed = nn.Embedding(self.vocab_size, self.model_dim, padding_idx=tokenizer.pad_token_id)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        # U-net design by with learnable skip connection weights for decoder layers
        self.num_layers = config.num_layers
        self.num_encoder_layers = config.num_layers // 2  # Half of the layers for encoder
        self.skip_weights = nn.Parameter(torch.ones(self.num_encoder_layers))

    def get_encoder_block_mask(
            self,
            input_ids: torch.Tensor,
            pos_ids: torch.Tensor,
            sliding_window_size: Optional[torch.Tensor]
    ):
        docs = (input_ids[pos_ids] == self.cls_id).cumsum(dim=0)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            mask = docs[q_idx] == docs[kv_idx]
            if sliding_window_size is not None:
                bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
                mask = mask & bidirectional_sliding_window_mask
            return mask

        S = len(input_ids)
        return create_block_mask(doc_mask_mod, None, None, S, S)

    def encoder_pass(self, x0, block_mask, pos_ids):
        skip_connections = []
        value_residuals = []  # map value residuals from first 1/4th of layers to last 1/4th of layers
        x = x0
        for i in range(self.num_layers):
            if i >= self.num_encoder_layers:
                x = x + self.skip_weights[i - self.num_encoder_layers] * skip_connections.pop()
            vr = value_residuals.pop(0) if i >= self.num_layers * 3 // 4 else None
            x, v = self.blocks[i](x, vr, x0, block_mask, pos_ids)
            if i < self.num_layers // 4:
                value_residuals.append(v)
            if i < self.num_encoder_layers:
                skip_connections.append(x)
        return x

    def forward(self, input_ids, sliding_window_size: torch.Tensor = None, pos_ids: torch.Tensor = None):
        input_ids = input_ids.flatten()
        if pos_ids is None:
            pos_ids = torch.arange(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        block_mask = self.get_encoder_block_mask(input_ids, pos_ids, sliding_window_size)
        x0 = norm(self.embed(input_ids[None]).bfloat16())
        return norm(self.encoder_pass(x0, block_mask, pos_ids))


class KBERTHead(CastedLinear):
    def __init__(self, model_dim: int, vocab_size: int, softcap: Optional[int] = None):
        super().__init__(model_dim, vocab_size)
        self.softcap = softcap

    def forward(self, x):
        x = super().forward(x)  # CastedLinear forward
        if self.softcap is not None:
            x = self.softcap * torch.tanh(x / self.softcap)
        return x.float()


class KBERTForMaskedLM(PreTrainedModel):
    config_class = ModelConfig
    _tied_weights_keys = ["lm_head.output_head.weight", "encoder.embed.weight"]

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_uri)
        self.masker = MLMMasker(tokenizer)
        self.encoder = KBERTModel(config, tokenizer)
        self.vocab_size = self.encoder.vocab_size
        self.lm_head = KBERTHead(config.model_dim, self.vocab_size, softcap=config.logit_softcap)

        self.encoder.embed.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            sliding_window_size: torch.Tensor,
            mask_prob: torch.Tensor,
            keep_replace_prob: torch.Tensor) -> torch.Tensor:
        input_ids, labels = self.masker(input_ids, labels, mask_prob, keep_replace_prob)
        last_hs = self.encoder(input_ids, sliding_window_size)
        logits = self.lm_head(last_hs)
        return F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())


class KBERTForSequenceClassification(PreTrainedModel):
    config_class = SequenceClassificationModelConfig

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_uri)
        self.bos_id = tokenizer.cls_token_id
        self.num_labels = config.num_labels
        self.encoder = KBERTModel(config, tokenizer)
        self.classifier_dropout = nn.Dropout(p=config.head_dropout)
        self.classifier_head = KBERTHead(config.model_dim, config.num_labels)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        last_hs = self.encoder(input_ids)
        pooled = last_hs[:, input_ids == self.bos_id, :]  # filter last_hs, only considering cls token outputs
        logits = self.classifier_head(self.classifier_dropout(pooled))
        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1).long())
        if return_logits:
            return loss, logits
        return loss


class MLMMasker(nn.Module):
    def __init__(self, tokenizer):
        """ELECTRA-style MLM objective, replacing mask_prob with mask, and randomly replacing keep_replace_prob"""
        super().__init__()
        self.mask_token_id = tokenizer.mask_token_id
        standard_tokens = [tok_id for tok_id in tokenizer.vocab.values() if tok_id not in tokenizer.all_special_ids]
        self.standard_tokens = nn.Buffer(torch.tensor(standard_tokens, dtype=torch.int32), persistent=False)
        self.special_tokens = nn.Buffer(torch.tensor(tokenizer.all_special_ids, dtype=torch.int32), persistent=False)

    def __call__(
            self, input_ids: torch.Tensor, labels: torch.Tensor, mask_prob: torch.Tensor, keep_replace_prob: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # include mlm_prob tokens in MLM objective
        mlm_prob = mask_prob + 2 * keep_replace_prob
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
