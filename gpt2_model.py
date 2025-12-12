import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2_config import GPT2Config


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_ctx, config.n_ctx))
            .view(1, 1, config.n_ctx, config.n_ctx),
            persistent=False,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        x = x.view(B, T, self.n_head, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, n_head, T, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, n_head * head_dim)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, C = x.size()

        qkv = self.c_attn(x)  
        q, k, v = qkv.split(C, dim=2)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = self.bias[:, :, :T, : k.size(-2)]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        attn_out, present = self.attn(self.ln_1(x), layer_past, use_cache)
        x = x + attn_out

        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out

        return x, present


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        self._residual_weight_scaling()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _residual_weight_scaling(self):
        """
        Scale residual branches by 1/sqrt(N) where N is number of residual layers.
        (Matches the description in the paper.)
        """
        n_layer = self.config.n_layer
        scale = 1.0 / math.sqrt(2.0 * n_layer)  

        for block in self.h:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        use_cache: bool = False,
    ):
        """
        input_ids: (B, T)
        labels: (B, T) or None
        """
        device = input_ids.device
        B, T = input_ids.shape
        assert T <= self.config.n_ctx, "Sequence length > model context size"

        if past_key_values is None:
            past_key_values = [None] * self.config.n_layer

        position_ids = torch.arange(0, T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)  

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        presents = [] if use_cache else None

        for block, layer_past in zip(self.h, past_key_values):
            hidden_states, present = block(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if use_cache:
            return logits, loss, tuple(presents)
        else:
            return logits, loss