import torch
from torchvision.models import quantization as qm

from .data import NUM_CLASSES


def build(arch: str, dropout: float = 0.0, hidden: int = 0, act: str = "relu") -> torch.nn.Module:
    """Quantizable torchvision models. QuantStub/DeQuantStub are identities in float,
    so the same class is trained in FP32/AMP and later quantized.
    dropout / hidden only change the ResNet-18 classifier (hyperparameter study)."""
    if arch == "resnet18":
        if act != "relu":
            # the quantizable block fuses add+ReLU, so other activations use the plain torchvision ResNet-18
            # (identical in float otherwise) with every ReLU module swapped
            from torchvision.models import resnet18
            m = resnet18(weights=None, num_classes=NUM_CLASSES)
            A = {"silu": torch.nn.SiLU, "gelu": torch.nn.GELU}[act]
            for mod in list(m.modules()):
                for n, ch in mod.named_children():
                    if isinstance(ch, torch.nn.ReLU):
                        setattr(mod, n, A())
        else:
            m = qm.resnet18(weights=None, quantize=False, num_classes=NUM_CLASSES)
        if dropout or hidden:
            d = m.fc.in_features
            layers = [torch.nn.Linear(d, hidden), torch.nn.ReLU(inplace=True)] if hidden else []
            m.fc = torch.nn.Sequential(*layers, torch.nn.Dropout(dropout), torch.nn.Linear(hidden or d, NUM_CLASSES))
        return m
    if arch == "mbv2_050":
        return qm.mobilenet_v2(weights=None, quantize=False, num_classes=NUM_CLASSES, width_mult=0.5)
    raise ValueError(arch)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
