"""Runtime environment detection and BF16/FP32 safety checks for Colab and local runs."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RuntimeEnv:
    platform: str
    root_dir: Path
    device: torch.device
    frozen_dtype: torch.dtype
    trainable_dtype: torch.dtype
    autocast_dtype: torch.dtype
    gpu_name: str | None
    vram_gb: float | None
    compute_capability: tuple[int, int] | None
    bf16_hardware: bool
    bf16_supported: bool


def detect_platform() -> tuple[str, Path]:
    if Path("/content").exists() and ("COLAB_RELEASE_TAG" in os.environ or Path("/content/sample_data").exists()):
        return "colab", Path("/content")
    return "local", Path.cwd()


def _test_bf16_ops(device: torch.device) -> bool:
    """Verify BF16 matmul, conv1d, SDPA, and backward pass run cleanly on this GPU."""
    if device.type != "cuda":
        return False
    try:
        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = x @ w

        # ShortConv-like 1D conv
        conv_w = torch.randn(64, 64, 3, device=device, dtype=torch.bfloat16, requires_grad=True)
        c = F.conv1d(y.transpose(1, 2), conv_w, padding=1).transpose(1, 2)

        # GQA / SDPA-like attention
        q = c.view(2, 16, 4, 16).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, q, q, is_causal=True)

        loss = attn.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        return (
            torch.isfinite(loss).item()
            and x.grad is not None
            and torch.isfinite(x.grad).all().item()
        )
    except Exception as e:
        print(f"BF16 op test failed ({type(e).__name__}: {e}); falling back to FP32.")
        return False


def get_runtime_env(prefer_bf16: bool = True) -> RuntimeEnv:
    platform, root_dir = detect_platform()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024**3)
        cc = (props.major, props.minor)
        bf16_hw = cc[0] >= 8
        bf16_ok = _test_bf16_ops(device) if prefer_bf16 else False
        # Never use FP16: LFM3 ships in BF16 and can overflow in FP16.
        # Keep trainable master weights in FP32 so small AdamW updates do not underflow.
        frozen_dtype = torch.bfloat16 if bf16_ok else torch.float32
        autocast_dtype = torch.bfloat16 if bf16_ok else torch.float32
        trainable_dtype = torch.float32
    else:
        device = torch.device("cpu")
        gpu_name = None
        vram_gb = None
        cc = None
        bf16_hw = False
        bf16_ok = False
        frozen_dtype = torch.float32
        autocast_dtype = torch.float32
        trainable_dtype = torch.float32

    return RuntimeEnv(
        platform=platform,
        root_dir=root_dir,
        device=device,
        frozen_dtype=frozen_dtype,
        trainable_dtype=trainable_dtype,
        autocast_dtype=autocast_dtype,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        compute_capability=cc,
        bf16_hardware=bf16_hw,
        bf16_supported=bf16_ok,
    )


def print_env(env: RuntimeEnv | None = None) -> RuntimeEnv:
    if env is None:
        env = get_runtime_env()

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    torch_ver = torch.__version__

    print(f"Platform:     {env.platform} (root={env.root_dir})")
    print(f"Python:       {py_ver} {'[OK]' if py_ok else '[WARN: need >= 3.11]'}")
    print(f"PyTorch:      {torch_ver}")

    try:
        import transformers

        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: not installed")

    if env.gpu_name is not None and env.compute_capability is not None and env.vram_gb is not None:
        cc_str = f"{env.compute_capability[0]}.{env.compute_capability[1]}"
        hw_note = "native BF16" if env.bf16_hardware else "no native BF16 hardware (emulated or FP32 fallback)"
        print(f"GPU:          {env.gpu_name} | {env.vram_gb:.1f} GB | CC {cc_str} ({hw_note})")
        print(
            f"Dtypes:       frozen={env.frozen_dtype}, autocast={env.autocast_dtype}, "
            f"trainable={env.trainable_dtype} "
            f"(BF16 ops+bwd test: {'PASS' if env.bf16_supported else 'FAIL -> FP32 fallback'})"
        )
    else:
        print("GPU:          None (CPU only)")
        print(f"Dtypes:       frozen={env.frozen_dtype}, autocast={env.autocast_dtype}, trainable={env.trainable_dtype}")

    return env


if __name__ == "__main__":
    print_env()
