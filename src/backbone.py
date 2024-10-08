# transformer.py

import math
import torch
import torch.nn as nn
from functools import partial
""" 
The transformer structure
The branch is in the "TransformerMUGS"
the DropPath
the MLP
the ATTENTION
the encoder block done
"""
def trunc_normal_(tensor, mean=0., std=1.):
    # Truncated normal initialization
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class Mlp(nn.Module): 
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class Attention(nn.Module):
    # Attention module in Transformer with dropout
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape

        qkv = self.qkv(x)  # (B, N, 3 * C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn_weights = attn
        attn = self.attn_dropout(attn)  # Dropout in attention weights

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj_dropout(self.proj(x)) 

        if return_attention:
            return x, attn_weights
        else:
            return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0., attn_dropout=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, dropout=dropout)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.dropout(y)
            x = x + self.dropout(self.mlp(self.norm2(x)))
            return x, attn_weights
        else:
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.mlp(self.norm2(x)))
            return x

class TimeSeriesTransformer(nn.Module):
    # Transformer Model
    def __init__(self, 
                 input_dim=1, 
                 embed_dim=128, 
                 depth=4, 
                 num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 dropout=0., 
                 attn_dropout=0., 
                 num_classes=0,
                 num_relation_blocks=0, 
                 max_len=5000, 
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Initialize
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder layers
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Relation blocks for local group supervision
        self.num_relation_blocks = num_relation_blocks
        if num_relation_blocks > 0:
            self.relation_blocks = nn.ModuleList([
                TransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                ) for _ in range(num_relation_blocks)
            ])

    def forward(self, x, return_all=False, local_group_memory_inputs=None):
        B, N, _ = x.shape
        x = self.input_projection(x)  # Shape: (B, N, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (B, N+1, embed_dim)
        
        # Add positional encodings
        pos_embed = self.pos_embed[:, :N+1, :]
        x = x + pos_embed  # Shape: (B, N+1, embed_dim)
        x = self.pos_drop(x)

        # Pass through Transformer encoder layers
        for blk in self.blocks:
            x = blk(x)

        # Process with relation blocks if applicable
        if self.num_relation_blocks > 0:
            mem = local_group_memory_inputs.get("mem")
            if mem is not None:
                m, _ = mem(x.mean(1))
                rx = torch.cat((x.mean(1).unsqueeze(1), m), dim=1)
            else:
                rx = x.mean(1).unsqueeze(1)  # Use the mean of the sequence as class token
            for blk in self.relation_blocks:
                rx = blk(rx)
            relation_out = self.norm(rx[:, 0])
        else:
            relation_out = self.norm(x.mean(1))

        x = self.norm(x)
        if self.num_classes > 0:
            logits = self.head(x[:, 0])
            return logits
        elif return_all:
            return x, relation_out
        else:
            return x[:, 0], relation_out

    def forward_knn(self, x):
        # Forward pass for k-NN evaluation
        B, N, _ = x.shape
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.pos_embed[:, :N+1, :]
        x = x + pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        B, N, _ = x.shape
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.pos_embed[:, :N+1, :]
        x = x + pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, attn = blk(x, return_attention=True)
                return attn

    def get_intermediate_layers(self, x, n=1):
        B, N, _ = x.shape
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.pos_embed[:, :N+1, :]
        x = x + pos_embed
        x = self.pos_drop(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

def timeseries_transformer_tiny(**kwargs):
    model = TimeSeriesTransformer(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model

def timeseries_transformer_small(**kwargs):
    model = TimeSeriesTransformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model

def timeseries_transformer_base(**kwargs):
    model = TimeSeriesTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model

def timeseries_transformer_large(**kwargs):
    model = TimeSeriesTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model
