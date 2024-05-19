import torch 
import torch.nn as nn 
from asyncio.log import logger

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
