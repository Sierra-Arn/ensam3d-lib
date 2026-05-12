# src/ensam3d_inference/core/utils.py
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from timm.layers import LayerNormFp32
from ..shared import DeviceType, config


def normalize_state_dict_keys(
    state_dict: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """
    Normalize checkpoint parameter keys to match the current model
    architecture.

    Flattens nested checkpoint dictionaries containing a state_dict
    entry, removes common distributed training prefixes, filters
    non-tensor entries, and remaps legacy parameter names to the
    attribute names expected by the current implementation.

    Parameters
    ----------
    state_dict : dict[str, Any]
        Checkpoint dictionary containing either raw parameter tensors or
        a nested state_dict mapping.

    Returns
    -------
    dict[str, torch.Tensor]
        Flat parameter dictionary mapping normalized parameter names to
        tensors suitable for torch.nn.Module.load_state_dict.
    """
    if "state_dict" in state_dict:
        raw = state_dict["state_dict"]
    else:
        raw = state_dict

    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue

        new_key = key
        for prefix in ("model.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        new_key = _remap_key(new_key)
        out[new_key] = value

    return out


def _remap_key(key: str) -> str:
    """
    Remap a single checkpoint parameter key from legacy naming schemes
    to the current architecture.

    Applies prefix-level replacements followed by substring-level
    replacements using the compatibility mappings defined in
    pipe_config.

    Parameters
    ----------
    key : str
        Original parameter key from the checkpoint.

    Returns
    -------
    str
        Remapped parameter key matching the current module attribute
        names.
    """
    for old, new in config.checkpoint_legacy_prefixes:
        if key.startswith(old):
            key = new + key[len(old):]
            break

    for old, new in config.checkpoint_legacy_substrings:
        key = key.replace(old, new)

    return key


def load_weights_into_module(
    module: nn.Module,
    checkpoint_path: Path,
    map_location: DeviceType = DeviceType.CPU
) -> tuple[list[str], list[str]]:
    """
    Load checkpoint weights into an already constructed module.

    The checkpoint is normalized to match the current architecture
    naming scheme before loading. Missing and unexpected keys are
    returned for compatibility inspection. If the module defines
    refresh_inference_precision, it is called after loading to restore
    inference-time dtype conversions.

    Parameters
    ----------
    module : nn.Module
        Module instance receiving the checkpoint weights.
    checkpoint_path : Path
        Filesystem path to the checkpoint file.
    map_location : DeviceType, optional
        Device used for initial tensor storage during checkpoint
        loading. Default is DeviceType.CPU.

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple containing missing_keys and unexpected_keys as reported by
        torch.nn.Module.load_state_dict with strict=False.
    """
    blob = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    flat = normalize_state_dict_keys(blob)
    
    missing, unexpected = module.load_state_dict(flat, strict=False)
    refresh = getattr(module, "refresh_inference_precision", None)
    if callable(refresh):
        refresh()

    return missing, unexpected


def resolve_model_path(model_path: str | Path) -> Path:
    """
    Resolve a local model path or download the model from HuggingFace.

    If model_path exists on the local filesystem, it is returned
    directly. Otherwise, the value is treated as a HuggingFace
    repository ID and downloaded using snapshot_download.

    Parameters
    ----------
    model_path : str or Path
        Local filesystem path or HuggingFace repository identifier.

    Returns
    -------
    Path
        Local directory containing the resolved model files.
    """
    path = Path(model_path)
    if path.exists():
        return path
    
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(model_path)))


def cast_floating_params_buffers_bf16(module: nn.Module,) -> None:
    """
    Cast floating-point parameters and buffers to the configured
    inference compute dtype.

    Floating tensors belonging to LayerNormFp32 modules and tensors
    inside the TorchScript MHR submodule are excluded from casting to
    preserve numerical stability and TorchScript compatibility. The
    conversion is performed in-place without tracking gradients.

    Parameters
    ----------
    module : nn.Module
        Module whose floating-point parameters and buffers are cast
        in-place.
    """
    def _parent(name: str) -> nn.Module | None:
        if "." not in name:
            return None
        path = name.rsplit(".", 1)[0]
        try:
            return module.get_submodule(path)
        except AttributeError:
            return None

    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.startswith(config.mhr_skip_prefix):
                continue
            if isinstance(_parent(name), LayerNormFp32):
                continue
            if p.is_floating_point():
                p.data = p.data.to(config.core_compute_dtype)

        for name, t in module.named_buffers():
            if name.startswith(config.mhr_skip_prefix):
                continue
            if isinstance(_parent(name), LayerNormFp32):
                continue
            if t.is_floating_point():
                t.data = t.data.to(config.core_compute_dtype)