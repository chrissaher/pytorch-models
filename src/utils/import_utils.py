import os
import importlib.util
import importlib.metadata as importlib_metadata
from types import ModuleType
from itertools import chain
from typing import Any

from packaging import version


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

_torch_version = "N/A"
if USE_TORCH in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    _torch_available = False
    print("PYTORCH IS NOT ENABLED!!!!")

torch_version = None
if _torch_available:
    torch_version = version.parse(importlib_metadata.version("torch"))


def is_torch_available():
    return _torch_available


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    # And transformers.utils.import_utils._LazyModule
    # https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py

    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key

        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name__ = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
