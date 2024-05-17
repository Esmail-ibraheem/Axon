from dataclasses import dataclass
import math
import torch 
import torch.nn as nn
# from torch import nn as neural_network

@dataclass
class Model_Arguments:
    source_vocab_size: int
    target_vocab_size: int 
    source_sequence_length: int 
    target_sequence_length: int 
    d_model: int = 512 
    Layers: int = 6
    heads: int = 8 
    dropout: float = 0.1 
    d_ff: int = 2048 

class InputEmbeddingsLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(sequence_length, d_model)
        Position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        deviation_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        PE[:, 0::2] = torch.sin(Position * deviation_term)
        PE[:, 1::2] = torch.cos(Position * deviation_term)
        PE = PE.unsqueeze(0)
        self.register_buffer("PE", PE)
    def forward(self, x):
        x = x + (self.PE[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class NormalizationLayer(nn.Module):
    def __init__(self, Epslone: float = 10**-6) -> None:
        super().__init__()
        self.Epslone = Epslone
        self.Alpha = nn.Parameter(torch.ones(1))
        self.Bias = nn.Parameter(torch.ones(1))
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.Alpha * (x - mean) / (std + self.Epslone) + self.Bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.Linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.Linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.Linear_2(self.dropout(torch.relu(self.Linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads 
        assert d_model % heads == 0, "d_model is not divisable by heads"

        self.d_k = d_model // heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def Attention(Query, Key, Value, mask, dropout: nn.Module):
        d_k = Query.shape[-1]

        self_attention_score = (Query @ Key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            self_attention_score.masked_fill_(mask == 0, -1e9)
        self_attention_score = self_attention_score.softmax(dim = -1)

        if dropout is not None:
            self_attention_score = dropout(self_attention_score)
        return self_attention_score @ Value
    def forward(self, query, key, value, mask):
        Query = self.W_Q(query)
        Key = self.W_K(key)
        Value = self.W_V(value)

        Query = Query.view(Query.shape[0], Query.shape[1], self.heads, self.d_k).transpose(1,2)
        Key = Key.view(Key.shape[0], Key.shape[1], self.heads, self.d_k).transpose(1,2)
        Value = Value.view(Value.shape[0], Value.shape[1], self.heads, self.d_k).transpose(1,2)

        x, self.self_attention_score = MultiHeadAttentionBlock.Attention(Query, Key, Value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)
        return self.W_O(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = NormalizationLayer()
    def forward(self, x, subLayer):
        return x + self.dropout(subLayer(self.normalization(x)))

# Building the encoder block 
class EncoderBlock(nn.Module):
    def __init__(self, encoder_self_attention_block: MultiHeadAttentionBlock, encoder_feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.encoder_self_attention_block = encoder_self_attention_block
        self.encoder_feed_forward_block = encoder_feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, source_mask):
        x = self.residual_connection[0](x, lambda x: self.encoder_self_attention_block(x, x, x, source_mask))
        x = self.residual_connection[1](x, self.encoder_feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, Layers: nn.ModuleList) -> None:
        super().__init__()
        self.Layers = Layers
        self.normalization = NormalizationLayer()
    def forward(self, x, source_mask):
        for layer in self.Layers:
            x = layer(x, source_mask)
        return self.normalization(x)

# Building the decoder block 
class DecoderBlock(nn.Module):
    def __init__(self, decoder_self_attention_block: MultiHeadAttentionBlock, decoder_cross_attention_block: MultiHeadAttentionBlock, decoder_feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.decoder_self_attention_block = decoder_self_attention_block
        self.decoder_cross_attention_block = decoder_cross_attention_block
        self.decoder_feed_forward_block = decoder_feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self, x, Encoder_output, source_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.decoder_self_attention_block(x, x, x, source_mask))
        x = self.residual_connection[1](x, lambda x: self.decoder_cross_attention_block(x, Encoder_output, Encoder_output, target_mask))
        x = self.residual_connection[2](x, self.decoder_feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, Layers: nn.ModuleList) -> None:
        super().__init__()
        self.Layers = Layers
        self.normalization = NormalizationLayer()
    def forward(self, x, Encoder_output, source_mask, target_mask):
        for layer in self.Layers:
            x = layer(x, Encoder_output, source_mask, target_mask)
        return self.normalization(x)

class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.Linear = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return self.Linear(x)

# Building the transformer block 
class TransformerBlock(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbeddingsLayer, target_embedding: InputEmbeddingsLayer, source_position: PositionalEncodingLayer, target_position: PositionalEncodingLayer, Linear: LinearLayer) -> None:
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.targ_position = target_position
        self.Linear = Linear 
    def encode(self, source_language, source_mask):
        source_language = self.source_embedding(source_language)
        source_language = self.source_position(source_language)
        return self.encoder(source_language, source_mask)
    def decode(self, Encoder_output, source_mask, target_language, target_mask):
        target_language = self.target_embedding(target_language)
        target_language = self.targ_position(target_language)
        return self.decoder(target_language, Encoder_output, source_mask, target_mask)
    def linear(self, x):
        return self.Linear(x)

# Transformer model skeletone 
def Transformer_model(Args: Model_Arguments)->TransformerBlock:

    source_embedding = InputEmbeddingsLayer(Args.d_model, Args.source_vocab_size)
    source_position = PositionalEncodingLayer(Args.d_model, Args.source_sequence_length, Args.dropout)

    target_embedding = InputEmbeddingsLayer(Args.d_model, Args.target_vocab_size)
    target_position = PositionalEncodingLayer(Args.d_model, Args.target_sequence_length, Args.dropout)

    Encoder_Blocks = []
    for _ in range(Args.Layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(Args.d_model, Args.heads, Args.dropout)
        encoder_feed_forward_block = FeedForwardBlock(Args.d_model, Args.d_ff, Args.dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, Args.dropout)
        Encoder_Blocks.append(encoder_block)
    
    Decoder_Blocks = []
    for _ in range(Args.Layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(Args.d_model, Args.heads, Args.dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(Args.d_model, Args.heads, Args.dropout)
        decoder_feed_forward_block = FeedForwardBlock(Args.d_model, Args.d_ff, Args.dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, Args.dropout)
        Decoder_Blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(Encoder_Blocks))
    decoder = Decoder(nn.ModuleList(Decoder_Blocks))

    linear = LinearLayer(Args.d_model, Args.target_vocab_size)

    Transformer = TransformerBlock(encoder, decoder, source_embedding, target_embedding, source_position, target_position, linear)

    for t in Transformer.parameters():
        if t.dim() > 1:
            nn.init.xavier_uniform(t)
    return Transformer 