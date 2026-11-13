
"""VisionPort - Environment Check"""

import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

rows = []


def log(key, value):
    rows.append((key, str(value)))
    print(f"{key:<30}{value}")


def pkg(name):
    try:
        return version(name)
    except Exception:
        return "NOT INSTALLED"


# 1. System and Python
log("Date", datetime.now().strftime("%Y-%m-%d %H:%M"))
log("Operating system", platform.platform())
log("Python version", sys.version.split()[0])
log("Python interpreter", sys.executable)

# 2. Package versions
for name in [
    "torch", "torchvision", "ultralytics", "numpy",
    "pillow", "fastapi", "uvicorn", "streamlit", "pandas"
]:
    log(f"pkg: {name}", pkg(name))

# 3. NVIDIA driver
if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    log("NVIDIA driver", out.stdout.strip() or "Unreadable")
else:
    log("NVIDIA driver", "Not found")

# 4. PyTorch and CUDA
try:
    import torch

    log("PyTorch CUDA build", torch.version.cuda or "CPU-only")
    log("CUDA available", torch.cuda.is_available())

    if torch.backends.cudnn.is_available():
        log("cuDNN version", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        log("GPU count", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log(f"GPU {i}", p.name)

        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        log("GPU matmul test", f"PASS ({y.device})")
    else:
        log("GPU matmul test", "SKIPPED")
except Exception as e:
    log("PyTorch error", f"{type(e).__name__}: {e}")

# 5. Save report
lines = [
    "# VisionPort Environment Report",
    "",
    "| Item | Value |",
    "|---|---|"
]
lines += [f"| {k} | {v} |" for k, v in rows]

Path("env_report.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

print("\nSaved: env_report.md")
