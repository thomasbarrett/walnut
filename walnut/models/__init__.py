"""Model definitions and the architecture registry.

Each supported architecture lives in its own module (e.g. `qwen3_5`) and is
registered here by the string Hugging Face reports in ``config.architectures``.
`resolve_model_class` maps a loaded config to the right class; the engine uses
it to instantiate the model before loading weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from walnut.models.qwen3_5 import Qwen3_5ForConditionalGeneration

if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig

#: Maps ``config.architectures[0]`` to the walnut model class implementing it.
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGeneration,
}


def resolve_model_class(config: PretrainedConfig) -> type[nn.Module]:
    """Return the walnut model class for ``config``'s architecture.

    Raises ``ValueError`` if none of the config's declared architectures are
    registered.
    """
    architectures = getattr(config, "architectures", None) or []
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"no walnut model registered for architectures {architectures!r}; "
        f"known: {sorted(_MODEL_REGISTRY)}"
    )


__all__ = ["Qwen3_5ForConditionalGeneration", "resolve_model_class"]
