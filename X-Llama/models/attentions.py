from asyncio.log import logger
from typing import Optional, Tuple

import torch
import torch.nn as nn 
from torch.nn import functional as F 

import math 

from transformers.cache_utils import Cache

from model import (
    LlamaRotaryEmbeddings, 
    LlamaLinearScalingRotaryEmbeddings, 
    LlamaDynamicNTKScalingRotaryEmbeddings, 
    repeat_kv, 
    apply_rotary_embeddings
)
from config import LlamaConfig

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
    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

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

def MultiHeadAttention():
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

def MultiQueryAttention():
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