import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging 

from config import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(seqlens_in_batch.flatten(), as_tuple=False).flatten()
    max_seqlens_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1,0))
    return (
        indices,
        cu_seqlens,
        max_seqlens_in_batch
    )

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_eps = eps 
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype 
        hidden_states = hidden_states.to(torch.int32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance, self.variance_eps)
        return self.weight * hidden_states.to(input_dtype)

class LlamaFixedRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps 
    
    def _norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)
ALL_LAYERNORM_LAYERS.append(LlamaFixedRMSNorm)

class LlamaRotaryEmbeddings(nn.Module):
    def __init__(self, dim, max_position_embddings=2048, base=10000, device=None, scaling_factor=1.0) -> None:
        super().__init__()
        self.dim = dim 
        self.max_position_embeddings = max_position_embddings
        self.base = base 
        self.scaling_factor = scaling_factor
        self.inv_frequency = 1.0 / (self.base * (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_frequency", self.inv_frequency, persistent=False)
        self.max_seq_len_cached = max_position_embddings
        t = torch.arange(0, self.max_seq_len_cached, device=device)
        t = t / scaling_factor
        freqs = torch.outer(t, self.inv_frequency)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)
    
    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self.sin_cached
    
    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self.cos_cached

    @torch.no_grad
    def forward(self, x, position_ids):
        inv_frequency_expanded = self.inv_frequency[None, : , None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[None, :, None].float()
        device_dtype = x.device.dtype 
        device_dtype = device_dtype if isinstance(device_dtype, str) and device_dtype == "mps" else "cpu"
        with torch.autocast(device_type=device_dtype, enabled=False):
            freqs = (inv_frequency_expanded.float() @ position_ids_expanded.float())
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaLinearScalingRotaryEmbeddings(LlamaRotaryEmbeddings):
    def forward(self, x, position_ids):
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin 

class LlamaDynamicNTKScalingRotaryEmbeddings(LlamaRotaryEmbeddings):
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1 
        if seq_len > self.max_position_embeddings:
            base = self.base * (self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
            inv_frequency = 1.0 / (base * (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
            self.register_buffer("inv_frequency", inv_frequency, persistent=False)
            cos, sin = super().forward(x, position_ids)
        return cos, sin 

LLAMA_ROTARY_EMBEDDINGS_CLASSES = {
    "rotary": LlamaRotaryEmbeddings,
    "linear": LlamaLinearScalingRotaryEmbeddings,
    "dynamic": LlamaDynamicNTKScalingRotaryEmbeddings,
}

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float=10000.0):
    assert head_dim % 2 == 0 , "dimension must be divisable by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: int, device:str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = torch.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x:torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape 
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )
class LlamaScalableGroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_heads_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True 

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.query_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.key_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.value_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
    
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.emb_rotary = LlamaRotaryEmbeddings(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta
            )
        else :
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.emb_rotary = LlamaLinearScalingRotaryEmbeddings(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta
                )
            elif scaling_type == "dynamic":
                self.emb_rotary = LlamaDynamicNTKScalingRotaryEmbeddings(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta
                )
            else:
                raise ValueError(f"Unkown scaling type of RoPE {scaling_type}")
    
    def forward(
        self, 
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None ,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None, 
        attention_output: bool = False,
        position_cache: Optional[torch.LongTensor] = None 
    )->Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[Optional[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.query_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)

            key_slices = self.key_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.value_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        
        else:
            query_states = self.query_proj(hidden_states)
            key_states = self.key_proj(hidden_states)
            value_states = self.value_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.emb_rotary(value_states, position_ids)
        query_states, key_states = apply_rotary_embeddings(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_heads_groups)
        value_states = repeat_kv(value_states, self.num_key_value_heads_groups)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "position_cache": position_cache}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        attention_weights = torch.matmul(query_states, key_states.transpose(3,2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attention_weights = attention_weights + causal_mask
        
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_weights = nn.functional.dropout(attention_weights, p=self.config.attention_dropout, training=self.training)
        output_attention = torch.matmul(attention_weights, value_states)

        if output_attention.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attention_output.size()}"
            )
        output_attention = output_attention.transpose(1,2).contiguous()
        output_attention = output_attention.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 2:
            output_attention = output_attention.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            output_slices = self.output_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            output_attention = sum([F.linear(output_attention[i], output_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            output_attention = self.output_proj(output_attention)

        if not attention_output :
            attention_weights = None 

        return attention_weights, output_attention, past_key_value

class LlamaFixedGroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_heads_q = config.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.cache_k = torch.zeros((config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

class MultiHeadAttention(nn.Module):
    def MultiHeadAttention(self):
        d_model, batch, heads, key, value = 512, 32, 8, (512 // 8), (512 // 8)
        
        m = 5 # suppose we have already cached "m" tokens 
        previous_key = torch.rand(batch, heads, m, key)
        previous_value = torch.rand(batch, heads, m, value)

        X = torch.rand(batch, d_model) # query
        M = torch.rand(batch, d_model) # key and value 

        P_q = torch.rand(heads, d_model, key)
        P_k = torch.rand(heads, d_model, key)
        P_v = torch.rand(heads, d_model, value)
        P_o = torch.rand(heads, d_model, value)

        q = torch.einsum("bd,hdk->bhk", X, P_q)
        new_k = torch.concat([previous_key, torch.einsum("bd,hdk->bhk", M, P_k).unsqueeze(2)], axis=2)
        new_v = torch.concat([previous_value, torch.einsum("bd,hdk->bhk", M, P_v).unsqueeze(2)], axis=2)

        logits = torch.einsum("bhk,bhmk->bhm", q, new_k)
        weights = torch.softmax(logits, dim=-1)
        output = torch.einsum("bhm,bhmv->bhv", weights, new_v)
        y = torch.einsum("bhv,hdv->bd", output, P_o)

        return y, new_k, new_v


class MultiQueryAttention:
    def MultiQueryAttention(self):
        d_model, batch, heads, key, value = 512, 32, 8, (512 // 8), (512 // 8)
        
        m = 5 # suppose we have already cached "m" tokens 
        previous_key = torch.rand(batch, m, key)
        previous_value = torch.rand(batch,  m, value)

        X = torch.rand(batch, d_model) # query
        M = torch.rand(batch, d_model) # key and value 

        P_q = torch.rand(heads, d_model, key)
        P_k = torch.rand(d_model, key)
        P_v = torch.rand(d_model, value)
        P_o = torch.rand(heads, d_model, value)

        q = torch.einsum("bd,hdk->bhk", X, P_q)
        k = torch.concat([previous_key, torch.einsum("bd,dk->bk", M, P_k).unsqueeze(1)], axis=1)
        v = torch.concat([previous_value, torch.einsum("bd,dv->bv", M, P_v).unsqueeze(1)], axis=1)

        logits = torch.einsum("bhk,bmk->bhm", q, k)
        weights = torch.softmax(logits, dim=-1)
        output = torch.einsum("bhm,bhmv->bhv", weights, v)
        y = torch.einsum("bhv,hdv->bd", output, P_o)

        return y, k, v

LLAMA_ATTENTIONS_CLASSES = {
    "GQA": LlamaScalableGroupedQueryAttention,
    "MHA": MultiHeadAttention,
    "MQA": MultiQueryAttention,
}

class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * config.ffn_dim_multiplier)
        hidden_dim = config.multiple_of * ((hidden_dim * config.multiple_of - 1) // config.multiple_of)

        self.weight_1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.weight_2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.weight_3 = nn.Linear(config.dim, hidden_dim, bias=False)
    
    def forward(self, x:torch.Tensor):
        swish = F.silu(self.weight_1(x))
        x_V = self.weight_3(x)
        x = swish * x_V 
        x = self.weight_2(x)
        return x

class LlamaEncoderBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads

        self.attention = LlamaFixedGroupedQueryAttention(config)
        # self.self_attn = LLAMA_ATTENTIONS_CLASSES[config._attn_implementation](config=config, layer_idx=self.layer_idx)
        self.MLP = LlamaMLP(config)

        self.attention_norm = LlamaFixedRMSNorm(config.dim, eps=config.norm_eps)
        self.ff_norm = LlamaFixedRMSNorm(config.dim, eps=config.norm_eps)
    
    def forward(self, x:torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.MLP.forward(self.ff_norm(h))
        return out 

class Transformer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()

        assert config.vocab_size != -1, "Vocab size must be set"

        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, config.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaEncoderBlock(config))

        self.norm = LlamaFixedRMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.config.dim // self.config.n_heads, self.config.max_seq_len * 2, device=self.config.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        h = self.tok_embeddings(tokens)

        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output