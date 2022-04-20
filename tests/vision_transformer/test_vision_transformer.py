import inspect
import torch.nn as nn
import unittest
import unittest.mock as mock

# from models import VisionTransformerConfig

from models.vision_transformer import VisionTransformerConfig, VisionTransformer

from testing_utils import is_torch_available, torch_device, require_torch, slow
from ..test_modeling_common import floats_tensor


class VisionTransformerTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_classes=2,
        image_size=224,
        num_channels=3,
        patch_size=32,
        embedding_dim=1024,
        num_layers=8,
        attention_drop=0.,
        num_heads=8,
        mlp_hidden=2048,
        drop_rate=0.,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size=patch_size
        self.embedding_dim=embedding_dim
        self.num_layers=num_layers
        self.attention_drop=attention_drop
        self.num_heads=num_heads
        self.mlp_hidden=mlp_hidden
        self.drop_rate=drop_rate

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return VisionTransformerConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            attention_drop=self.attention_drop,
            num_heads=self.num_heads,
            mlp_hidden=self.mlp_hidden,
            drop_rate=self.drop_rate,
        )

    def create_and_check_model(self, config, pixel_values):
        model = VisionTransformer.fromconfig(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expeted last hidden states B, C
        self.parent.assertEqual(
            result.shape,
            (self.batch_size, self.num_classes)
        )


@require_torch
class VisionTransformerTest(unittest.TestCase):

    all_model_classes = [VisionTransformer] if is_torch_available() else []

    def setUp(self):
        self.model_tester = VisionTransformerTester(self)

    def test_config(self):
        self.create_and_test_config_common_properties()

    def create_and_test_config_common_properties(self):
        return

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs()

        for model_class in self.all_model_classes:
            model = model_class.fromconfig(config=config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["x"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs()

        for model_class in self.all_model_classes:
            model = model_class.fromconfig(config=config)
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.assertTrue(
                        torch.all(module.weight == 1),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                    self.assertTrue(
                        torch.all(module.bias == 0),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @unittest.skip(reason="Not implemented yet")
    def test_model_from_pretrained(self):
        pass
