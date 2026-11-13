"""Weight-only int8 loading for --int8: about half the memory of bf16, at slightly different pixels.

Qwen-Image-2.1's transformer (7.1B) and text encoder (8.8B) are 13 GB and 16 GB in bf16. With --int8
every large nn.Linear is swapped for an Int8Linear on the meta device, then the checkpoint streams in
one tensor at a time: a big weight is quantized on the device as it arrives (one scale per output row),
everything else is kept at its stored dtype. Nothing is written to disk; the same as minimax h3 image
studio's loader, from which this is copied.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

#: Rows at or below this go through torch's int8 kernel, which reads the int8 weight as is: a
#: token at a time when the prompt enhancer generates. Bigger batches dequantize and use bf16 matmul.
SMALL_BATCH = 8


class Int8Linear(nn.Module):
    """nn.Linear with an int8 weight and one bf16 scale per output row."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_buffer("qweight", torch.empty(self.out_features, self.in_features, dtype=torch.int8, device="meta"))
        self.register_buffer("scale", torch.empty(self.out_features, dtype=torch.bfloat16, device="meta"))
        self.bias = linear.bias

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor) -> None:
        # The scale is rounded to bf16 first, so dequantizing is exactly qweight * scale.
        scale = (weight.abs().amax(dim=1).float() / 127).clamp(min=1e-12).to(torch.bfloat16)
        self.qweight = (weight.float() / scale.float()[:, None]).round_().clamp_(-127, 127).to(torch.int8)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if x.numel() // self.in_features <= SMALL_BATCH and x.dtype == torch.bfloat16:
            out = torch._weight_int8pack_mm(x.reshape(-1, self.in_features), self.qweight, self.scale)
            out = out.reshape(*shape[:-1], self.out_features)
            return out if self.bias is None else out + self.bias.to(out.dtype)
        weight = self.qweight.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        return F.linear(x, weight, None if self.bias is None else self.bias.to(x.dtype))


def shards(directory: Path) -> list[Path]:
    """The safetensors files of a component directory, in index order."""
    index = sorted(directory.glob("*.safetensors.index.json"))
    if index:
        names = dict.fromkeys(json.loads(index[0].read_text())["weight_map"].values())
        return [directory / n for n in names]
    return sorted(directory.glob("*.safetensors"))


def load_int8(
    model: nn.Module,
    files: list[Path],
    device: str,
    quantize: Callable[[str, nn.Linear], bool],
    rename: Callable[[str], str | None] = lambda key: key,
    tensor_hook: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
) -> nn.Module:
    """Fill ``model`` (parameters on meta, as accelerate's init_empty_weights leaves them) from ``files``.

    ``quantize(name, linear)`` picks the Linears stored as int8. ``rename`` maps a checkpoint key to
    the model's name for it, or None to skip it. ``tensor_hook`` may transform a tensor on the device.
    """
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            name = f"{parent_name}.{child_name}" if parent_name else child_name
            if type(child) is nn.Linear and quantize(name, child):
                setattr(parent, child_name, Int8Linear(child))
    modules = dict(model.named_modules())

    from safetensors import safe_open

    for path in files:
        with safe_open(path, "pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 (safe_open has no __iter__)
                name = rename(key)
                if name is None:
                    continue
                module_name, _, attr = name.rpartition(".")
                module = modules.get(module_name)
                if module is None:
                    raise KeyError(f"{path.name}: {key} has no place in {type(model).__name__}")
                tensor = f.get_tensor(key).to(device)
                if tensor_hook:
                    tensor = tensor_hook(name, tensor)
                if isinstance(module, Int8Linear) and attr == "weight":
                    module.quantize(tensor)
                elif attr in module._parameters:
                    module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
                else:
                    module._buffers[attr] = tensor
                del tensor

    missing = [n for n, p in model.named_parameters() if p.is_meta]
    missing += [n for n, b in model.named_buffers() if b.is_meta]
    if missing:
        raise KeyError(f"{type(model).__name__}: not in the checkpoint: {', '.join(missing[:8])}…")
    # Buffers made in __init__ (rotary frequencies) were built on the CPU.
    for module in model.modules():
        for attr, buffer in module._buffers.items():
            if buffer is not None and buffer.device.type != torch.device(device).type:
                module._buffers[attr] = buffer.to(device)
    return model.eval().requires_grad_(False)


if __name__ == "__main__":
    torch.manual_seed(0)
    linear = nn.Linear(256, 64, bias=True).to(torch.bfloat16)
    q = Int8Linear(linear)
    q.quantize(linear.weight.data)
    for rows in (1, 4, 300):  # the int8 kernel and the dequantized matmul
        x = torch.randn(rows, 256, dtype=torch.bfloat16)
        ref, got = linear(x).float(), q(x).float()
        err = (got - ref).abs().max() / ref.abs().max()
        assert err < 0.02, (rows, err)
    print("ok")
