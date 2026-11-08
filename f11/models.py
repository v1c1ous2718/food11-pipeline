import torch
from torchvision.models import quantization as qm

from .data import NUM_CLASSES


def build(arch: str) -> torch.nn.Module:
    """Quantizable torchvision models. QuantStub/DeQuantStub are identities in float,
    so the same class is trained in FP32/AMP and later quantized."""
    if arch == "resnet18":
        return qm.resnet18(weights=None, quantize=False, num_classes=NUM_CLASSES)
    if arch == "mbv2_050":
        return qm.mobilenet_v2(weights=None, quantize=False, num_classes=NUM_CLASSES, width_mult=0.5)
    raise ValueError(arch)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
