""" Vision Transformer in PyTorch
arXiv report: https://arxiv.org/abs/2010.11929

This codes is based on:
* https://github.com/rwightman/pytorch-image-models
* https://github.com/Atcold/pytorch-Deep-Learning/blob/master/15-transformer.ipynb
* https://github.com/google-research/vision_transformer/blob/master/vit_jax/models.py

Also, code style is borrowd from prof. Philipp Krähenbühl classes
"""
import torch
import torch.nn as nn
from .configuration_vision_transformer import VisionTransformerConfig


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        num_channels: int,
        out_channels: int,
        num_heads: int = 8,
        drop_rate: float = 0.,
    ):
        super().__init__()
        assert out_channels % num_heads == 0

        self.W_q = torch.nn.Linear(num_channels, out_channels, bias=False)
        self.W_k = torch.nn.Linear(num_channels, out_channels, bias=False)
        self.W_v = torch.nn.Linear(num_channels, out_channels, bias=False)
        self.W_h = torch.nn.Linear(out_channels, out_channels)

        self.attention_drop = torch.nn.Dropout(drop_rate)
        self.proj_drop = torch.nn.Dropout(drop_rate)

        self.d_k = out_channels // num_heads

        self.num_heads = num_heads
        self.q_scale = self.d_k ** -0.5

    def scaled_dot_product(self, Q, K, V):
        batch_size = Q.size(0)

        Q = Q * self.q_scale
        scores = torch.matmul(Q, K.transpose(2, 3))

        A = torch.nn.Softmax(dim=-1)(scores)
        A = self.attention_drop(A)
        H = torch.matmul(A, V)

        return H


    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        Q = self.split_heads(self.W_q(x), batch_size)
        K = self.split_heads(self.W_k(x), batch_size)
        V = self.split_heads(self.W_v(x), batch_size)

        H = self.scaled_dot_product(Q, K, V)
        H = self.group_heads(H, batch_size)

        H = self.W_h(H)
        H = self.proj_drop(H)

        return H


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        out_channels: int,
        num_heads: int = 8,
        mlp_hidden: int = 2048,
        attention_drop: float = 0.,
        drop_rate: float = 0.,
    ):
        super().__init__()
        self.layernorm1 = torch.nn.LayerNorm(num_channels)
        self.layernorm2 = torch.nn.LayerNorm(out_channels)
        self.mha = MultiHeadAttention(num_channels, out_channels, num_heads, attention_drop)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(out_channels, mlp_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(drop_rate),
            torch.nn.Linear(mlp_hidden, out_channels),
            torch.nn.Dropout(drop_rate),
        )
        self.dropout = torch.nn.Dropout(drop_rate)

    def forward(self, x):
        identity = x
        z = self.layernorm1(x)
        z = self.mha(z)
        z = self.dropout(z) + identity
        identity = z
        z = self.layernorm2(z)
        z = self.mlp(z) + identity
        return z

class VisionTransformerPreTrainedModel():
    """
    An anstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """
    config_class = VisionTransformerConfig
    supports_gradient_checkpointing = False

    def __init__weights(self, module):
        if isinstance(moduke, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, VisionTransformer):
            module.gradient_checkpointing = value

class VisionTransformer(torch.nn.Module):
    def __init__(
            self,
            img_size: int,
            num_channels: int,
            num_classes: int,
            patch_size: int = 32,
            embedding_dim: int = 1024,
            num_layers: int = 8,
            attention_drop: float =0.,
            num_heads: int = 8,
            mlp_hidden: int = 2048,
            drop_rate: float = 0.
    ):
        super().__init__()

        assert img_size % patch_size == 0
        num_patches = img_size**2 // patch_size**2
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.embed_layer = torch.nn.Conv2d(num_channels,
                                           embedding_dim,
                                           kernel_size=patch_size,
                                           stride = patch_size)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        self.pos_dropout = torch.nn.Dropout(drop_rate)
        L = []
        c = embedding_dim
        layers = [1024] * num_layers
        for l in layers:
            L.append(TransformerEncoderBlock(c, l, attention_drop=attention_drop,
                                             num_heads=num_heads,
                                             mlp_hidden=mlp_hidden,
                                             drop_rate=drop_rate))
            c = l
        L.append(torch.nn.LayerNorm(c))
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, num_classes)


    @classmethod
    def fromconfig(cls, config:VisionTransformerConfig):
        kwargs = {
            "img_size": config.image_size,
            "num_channels": config.num_channels,
            "num_classes": config.num_classes,
            "patch_size": config.patch_size,
            "embedding_dim": config.embedding_dim,
            "num_layers": config.num_layers,
            "attention_drop": config.attention_drop,
            "num_heads": config.num_heads,
            "mlp_hidden": config.mlp_hidden,
            "drop_rate": config.drop_rate,
        }
        return cls(**kwargs)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        x = self.embed_layer(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_dropout(x)
        z = self.network(x)
        cls_token = z[:, 0]
        z = self.classifier(cls_token)
        return z
