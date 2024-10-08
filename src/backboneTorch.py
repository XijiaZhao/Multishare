import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # MLP module 
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

class TimeSeriesTransformer(nn.Module):
    # Transformer
    def __init__(self, 
                 input_dim=1, 
                 embed_dim=512, 
                 depth=4, 
                 num_heads=6, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 dropout=0.1, 
                 attn_dropout=0., 
                 num_classes=0,
                 num_relation_blocks=0, 
                 max_len=5000, 
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Initializations
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder layers 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='relu'
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Relation blocks for local group supervision
        self.num_relation_blocks = num_relation_blocks
        if num_relation_blocks > 0:
            self.relation_blocks = nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim, 
                        nhead=num_heads, 
                        dim_feedforward=int(embed_dim * mlp_ratio), 
                        dropout=dropout,
                        activation='relu'
                    ), num_layers=1
                ) for _ in range(num_relation_blocks)
            ])

    def forward(self, x, return_all=False, local_group_memory_inputs=None):
        B, N, _ = x.shape
        x = self.input_projection(x)  # Shape: (B, N, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (B, N+1, embed_dim)
        
        # Positional encodings
        pos_embed = self.pos_embed[:, :N+1, :]
        x = x + pos_embed  # Shape: (B, N+1, embed_dim)
        x = self.pos_drop(x)
        x = self.blocks(x)

        # Process with relation blocks
        if self.num_relation_blocks > 0:
            mem = local_group_memory_inputs.get("mem")
            if mem is not None:
                m, _ = mem(x.mean(1))
                rx = torch.cat((x.mean(1).unsqueeze(1), m), dim=1)
            else:
                rx = x.mean(1).unsqueeze(1)  # use the mean of the sequence as class token
            for blk in self.relation_blocks:
                rx = blk(rx)
            relation_out = self.norm(rx[:, 0])
            #print(relation_out.shape)
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