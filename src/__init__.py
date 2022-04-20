# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

from typing import TYPE_CHECKING
from .import_utils import is_torch_available


_import_structure = {
    "models.vision_transformer": ["VisionTransformerConfig"]
}

if is_torch_available():
    _import_structure["models.vision_transformer"].extend(
        [
            "VisionTransformer",
            "VisionTransformerPreTrainedModel",
        ]
    )

if TYPE_CHECKING:
    from .models.vision_transformer import VisionTransformerConfig

    if is_torch_available():
        from .models.vision_transformer import VisionTransformerPreTrainedModel, VisionTransformer

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
