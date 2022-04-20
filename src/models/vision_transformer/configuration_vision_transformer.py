""" ViT model configuration """


class VisionTransformerConfig():
    def __init__(
        self,
        image_size=224,
        num_channels=3,
        num_classes=2,
        patch_size=32,
        embedding_dim=1024,
        num_layers=8,
        attention_drop=0.,
        num_heads=8,
        mlp_hidden=2048,
        drop_rate=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.attention_drop = attention_drop
        self.num_heads = num_heads
        self.mlp_hidden = mlp_hidden
        self.drop_rate = drop_rate
