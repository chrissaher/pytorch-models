from typing import TYPE_CHECKING

from utils import _LazyModule, is_torch_available


_import_structure = {
    "configuration_vision_transformer": ["VisionTransformerConfig"],
}

if is_torch_available():
    _import_structure["modeling_vision_transformer"] = [
        "VisionTransformer",
        "VisionTransformerPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_vision_transformer import VisionTransformerConfig

    if is_torch_available():
        from .vision_transformer import VisionTransformer, VisionTransformerPreTrainedModel

else:
    # import sys
    #
    # sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
    from .configuration_vision_transformer import VisionTransformerConfig

    if is_torch_available():
        from .vision_transformer import VisionTransformer, VisionTransformerPreTrainedModel
