
"""
VisionPort - Model Artefact Lifecycle Inspector
================================================

Purpose
-------
Inspect the model files used in the VisionPort project and explain where each
file belongs in the deployment lifecycle:

    PyTorch checkpoint (.pt) -> ONNX model (.onnx) -> TensorRT engine (.engine)

This script can:
1. List model artefacts found in the project folder.
2. Inspect a YOLOv8 PyTorch checkpoint.
3. Report architecture, layers, parameters, classes, and checkpoint state.
4. Check whether ONNX and TensorRT engine files exist.
5. Report whether PyTorch can see an NVIDIA CUDA GPU.
6. Optionally export the checkpoint to ONNX.
7. Optionally build a TensorRT engine from an ONNX file using trtexec.

Important
---------
- Inspection is the main activity. It does not train the model or benchmark FPS.
- A CUDA-capable GPU does not automatically mean TensorRT is installed.
- TensorRT engine creation requires a compatible NVIDIA GPU, NVIDIA driver,
  TensorRT installation, and the `trtexec` command available in PATH.
- The engine is built for the environment/GPU on which it is compiled. It may
  need to be rebuilt for a different GPU or TensorRT environment.
- Keep the original .pt checkpoint. The .engine file is not a replacement for it.

Install basic dependencies:
    pip install ultralytics torch

Optional ONNX export dependencies:
    pip install onnx onnxruntime

Run examples:
    python inspect_artefacts.py
    python inspect_artefacts.py --weights models/yolov8n.pt
    python inspect_artefacts.py --weights models/yolov8n.pt --export-onnx
    python inspect_artefacts.py --onnx models/yolov8n.onnx --build-engine
    python inspect_artefacts.py --onnx models/yolov8n.onnx --build-engine --fp16

For a TensorRT build, make sure `trtexec` is installed and available in PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    "node_modules",
}


def format_size(number_of_bytes: int) -> str:
    """Return a human-readable file size."""
    size = float(number_of_bytes)

    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} GB"


def find_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Find files with the requested suffixes, excluding environment folders."""
    matches: list[Path] = []

    if not root.exists():
        return matches

    for folder, subfolders, filenames in os.walk(root):
        subfolders[:] = [
            name for name in subfolders if name not in EXCLUDED_DIRS
        ]

        for filename in filenames:
            path = Path(folder) / filename

            if path.suffix.lower() in suffixes:
                matches.append(path)

    return sorted(matches)


def print_file_inventory(root: Path) -> None:
    """Display the model artefact files currently present."""
    print("\n" + "=" * 72)
    print("1. ARTEFACT INVENTORY")
    print("=" * 72)

    artefacts = find_files(root, (".pt", ".onnx", ".engine", ".plan"))

    if not artefacts:
        print("No .pt, .onnx, .engine, or .plan files were found.")
        return

    descriptions = {
        ".pt": "PyTorch checkpoint: architecture/weights and possibly training state",
        ".onnx": "Portable model graph with weights",
        ".engine": "TensorRT compiled engine for a compatible NVIDIA environment",
        ".plan": "TensorRT serialized engine (another common extension)",
    }

    for path in artefacts:
        try:
            size = format_size(path.stat().st_size)
        except OSError:
            size = "size unavailable"

        description = descriptions.get(
            path.suffix.lower(),
            "Model artefact"
        )

        print(f"{str(path):48} {size:>10}   {description}")


def report_gpu_status() -> bool:
    """Report CUDA availability and visible GPU names."""
    print("\n" + "=" * 72)
    print("2. NVIDIA GPU / CUDA STATUS")
    print("=" * 72)

    if torch is None:
        print("PyTorch is not installed, so CUDA availability cannot be checked.")
        return False

    try:
        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            print("CUDA available: NO")
            print("PyTorch will use the CPU in this environment.")
            return False

        print("CUDA available: YES")
        print("Visible GPU count:", torch.cuda.device_count())

        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

        return True

    except Exception as exc:
        print("Could not query CUDA:", exc)
        return False


def load_checkpoint(checkpoint_path: Path) -> Any | None:
    """Load a PyTorch checkpoint for inspection."""
    if torch is None:
        print("PyTorch is not installed. Install it with: pip install torch")
        return None

    if not checkpoint_path.is_file():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    try:
        try:
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
            )

        return checkpoint

    except Exception as exc:
        print("Could not load the checkpoint for inspection.")
        print("Reason:", exc)
        print(
            "Check that the checkpoint is a trusted file and that the "
            "Ultralytics/PyTorch versions are compatible."
        )
        return None


def get_model_from_checkpoint(checkpoint: Any) -> Any | None:
    """Get the model object from common Ultralytics checkpoint keys."""
    if not isinstance(checkpoint, dict):
        return None

    model = checkpoint.get("ema")

    if model is None:
        model = checkpoint.get("model")

    return model


def inspect_checkpoint(checkpoint_path: Path) -> None:
    """Inspect checkpoint metadata, architecture, parameters, and class names."""
    print("\n" + "=" * 72)
    print(f"3. PYTORCH CHECKPOINT: {checkpoint_path}")
    print("=" * 72)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is None:
        return

    if isinstance(checkpoint, dict):
        print("Top-level contents of the checkpoint file:")

        for key, value in checkpoint.items():
            print(f"  {key:18} {type(value).__name__}")

        optimizer_present = checkpoint.get("optimizer") is not None

        print(
            "\nTraining state (optimizer) present:",
            "yes" if optimizer_present else "no"
        )

        if not optimizer_present:
            print("The checkpoint appears to be stripped of optimizer state.")
            print("It can still be used for inference, export, and fine-tuning.")
            print("It cannot restore the original optimizer state for an exact resume.")

    else:
        print("Checkpoint is not a dictionary; it may be a serialized model object.")

    model = get_model_from_checkpoint(checkpoint)

    if model is None:
        print("\nCould not locate a model object under the 'model' or 'ema' key.")
        print("Checkpoint metadata was displayed, but model details are unavailable.")
        return

    # Architecture / module count
    try:
        modules = list(model.modules())
        top_level_layers = getattr(model, "model", None)

        if top_level_layers is not None:
            try:
                layer_count = len(top_level_layers)
            except TypeError:
                layer_count = len(modules)
        else:
            layer_count = len(modules)

        print(
            f"\nArchitecture: {type(model).__name__}, "
            f"{layer_count} top-level layers"
        )

        if top_level_layers is not None:
            try:
                for index, layer in enumerate(top_level_layers):
                    if index < 3 or index >= max(3, layer_count - 1):
                        parameter_count = sum(
                            parameter.numel()
                            for parameter in layer.parameters()
                        )

                        print(
                            f"  layer {index:2}: "
                            f"{type(layer).__name__:16} "
                            f"{parameter_count:>12,} parameters"
                        )

                    elif index == 3 and layer_count > 5:
                        print("  ...")

            except TypeError:
                print("  Layer-by-layer listing is not available.")

    except Exception as exc:
        print("\nArchitecture inspection issue:", exc)

    # Parameters and tensor precision
    try:
        named_parameters = list(model.named_parameters())

        parameter_count = sum(
            parameter.numel()
            for _, parameter in named_parameters
        )

        tensor_count = len(named_parameters)

        print(
            f"\nWeights: {tensor_count} named parameter tensors, "
            f"{parameter_count:,} parameters"
        )

        if named_parameters:
            dtypes = sorted({
                str(parameter.dtype)
                for _, parameter in named_parameters
            })

            print(
                "Parameter dtype(s) after loading:",
                ", ".join(dtypes)
            )

            total_bytes = sum(
                parameter.numel() * parameter.element_size()
                for _, parameter in named_parameters
            )

            print(
                "Approx. parameter memory after loading:",
                format_size(total_bytes)
            )

            print("\nExample parameter tensors:")

            for name, parameter in named_parameters[:5]:
                print(
                    f"  {name:42} shape {tuple(parameter.shape)}"
                )

    except Exception as exc:
        print("\nCould not inspect model parameters:", exc)

    # Class names
    names = getattr(model, "names", None)

    if names is None:
        names = getattr(
            getattr(model, "model", None),
            "names",
            None
        )

    if names is not None:
        if isinstance(names, dict):
            class_names = list(names.values())
        else:
            class_names = list(names)

        print(f"\nClasses the model predicts: {len(class_names)}")
        print("Example classes:", class_names[:10])

    else:
        print("\nClass names were not found in the loaded model object.")

    print(
        "\nTeaching note: the checkpoint is the PyTorch source artefact. "
        "It contains the learned model parameters and model structure; "
        "training-state information may or may not be included."
    )


def inspect_onnx(onnx_path: Path) -> None:
    """Report whether an ONNX artefact exists."""
    print("\n" + "=" * 72)
    print("4. ONNX GRAPH")
    print("=" * 72)

    if onnx_path.is_file():
        print("ONNX file found:", onnx_path)
        print("File size:", format_size(onnx_path.stat().st_size))
        print("This is the portable graph hand-over format.")

        try:
            import onnx

            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            print("ONNX structural validation: PASSED")
            print(
                "Inputs:",
                [item.name for item in onnx_model.graph.input]
            )
            print(
                "Outputs:",
                [item.name for item in onnx_model.graph.output]
            )

        except ImportError:
            print("Install onnx to perform structural validation: pip install onnx")

        except Exception as exc:
            print("ONNX structural validation failed:", exc)

    else:
        print("No ONNX file found at:", onnx_path)
        print("Create one with:")
        print(
            "  python inspect_artefacts.py "
            "--weights models/yolov8n.pt --export-onnx"
        )


def inspect_tensorrt_engine(engine_path: Path) -> None:
    """Report whether a TensorRT engine exists."""
    print("\n" + "=" * 72)
    print("5. TENSORRT ENGINE")
    print("=" * 72)

    if engine_path.is_file():
        print("TensorRT engine found:", engine_path)
        print("File size:", format_size(engine_path.stat().st_size))
        print(
            "This is a compiled engine intended for a compatible "
            "TensorRT environment."
        )

    else:
        print("No TensorRT engine found at:", engine_path)
        print(
            "This is not automatically created just because an NVIDIA GPU is present."
        )
        print(
            "Build it explicitly from ONNX in a compatible TensorRT environment."
        )


def export_onnx(
    weights_path: Path,
    output_path: Path,
    image_size: int
) -> Path | None:
    """Export a YOLO checkpoint to ONNX using the Ultralytics Python API."""
    print("\n" + "=" * 72)
    print("OPTIONAL: EXPORT PYTORCH CHECKPOINT TO ONNX")
    print("=" * 72)

    if not weights_path.is_file():
        print("Cannot export: checkpoint not found:", weights_path)
        return None

    try:
        from ultralytics import YOLO

    except ImportError:
        print("Ultralytics is not installed.")
        print("Install it with: pip install ultralytics")
        return None

    try:
        model = YOLO(str(weights_path))

        exported = model.export(
            format="onnx",
            imgsz=image_size,
            dynamic=False,
        )

        exported_path = Path(exported)

        if exported_path.is_file():
            print("ONNX export successful:", exported_path)
            return exported_path

        if output_path.is_file():
            print("ONNX export successful:", output_path)
            return output_path

        print("Export command completed, but the output file was not located.")
        print("Returned value:", exported)
        return None

    except Exception as exc:
        print("ONNX export failed:", exc)
        print("Check that the required ONNX dependencies are installed.")
        return None


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    use_fp16: bool,
) -> Path | None:
    """Build a TensorRT engine by calling NVIDIA's trtexec utility."""
    print("\n" + "=" * 72)
    print("OPTIONAL: BUILD TENSORRT ENGINE")
    print("=" * 72)

    if not onnx_path.is_file():
        print("Cannot build engine: ONNX file not found:", onnx_path)
        print("Export or provide an ONNX file first.")
        return None

    if torch is None or not torch.cuda.is_available():
        print("PyTorch cannot detect a CUDA GPU in this environment.")
        print("TensorRT engine building has not been attempted.")
        return None

    trtexec = shutil.which("trtexec")

    if trtexec is None:
        print("The 'trtexec' command was not found in PATH.")
        print("Install/configure NVIDIA TensorRT and make trtexec available.")
        return None

    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]

    if use_fp16:
        command.append("--fp16")
        print("Requested precision: FP16 (if supported by the GPU and TensorRT).")

    else:
        print(
            "Requested precision: TensorRT default precision; "
            "FP16 was not requested."
        )

    print("\nCommand:")
    print(
        " ".join(
            f'"{part}"' if " " in part else part
            for part in command
        )
    )

    print("\nBuilding engine. This may take some time...")

    try:
        subprocess.run(command, check=True)

        if engine_path.is_file():
            print("\nTensorRT engine created:", engine_path)
            print("File size:", format_size(engine_path.stat().st_size))
            return engine_path

        print("\ntrtexec completed, but the expected engine file was not found.")
        return None

    except FileNotFoundError:
        print("Could not start trtexec. Check the TensorRT installation.")

    except subprocess.CalledProcessError as exc:
        print("TensorRT engine build failed.")
        print("Return code:", exc.returncode)
        print("Read the trtexec output above for the specific error.")

    except Exception as exc:
        print("Unexpected TensorRT build error:", exc)

    return None


def print_comparison() -> None:
    """Print the conceptual comparison used in the classroom."""
    print("\n" + "=" * 72)
    print("6. ARTEFACT COMPARISON")
    print("=" * 72)

    rows = [
        (
            "Checkpoint (.pt)",
            "Model structure + learned weights",
            "PyTorch / Python",
            "Training, fine-tuning, inference, export"
        ),
        (
            "ONNX (.onnx)",
            "Portable computational graph + weights",
            "ONNX Runtime or compatible compiler",
            "Model hand-over and conversion"
        ),
        (
            "TensorRT (.engine)",
            "Compiled and optimised execution engine",
            "Compatible NVIDIA GPU + TensorRT",
            "Optimised inference/serving"
        ),
    ]

    print(
        f"{'Artefact':22} {'Contains':36} "
        f"{'Needs':32} Used for"
    )

    print("-" * 120)

    for artefact, contains, needs, used_for in rows:
        print(
            f"{artefact:22} {contains:36} "
            f"{needs:32} {used_for}"
        )

    print("\nLifecycle:")
    print(
        "  PyTorch checkpoint (.pt)  ->  ONNX graph (.onnx)  "
        "->  TensorRT engine (.engine)"
    )

    print("\nRemember:")
    print("- .pt is retained as the source checkpoint.")
    print("- ONNX provides a portable graph representation.")
    print("- TensorRT compiles the graph for optimised inference on a compatible GPU.")
    print("- This script inspects artefacts; it does not measure inference FPS.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect VisionPort PyTorch, ONNX, and TensorRT model artefacts."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/yolov8n.pt"),
        help="Path to the PyTorch checkpoint (default: models/yolov8n.pt).",
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help=(
            "Path to an ONNX file. Default: same folder/name as "
            "checkpoint with .onnx."
        ),
    )

    parser.add_argument(
        "--engine",
        type=Path,
        default=None,
        help=(
            "Path to a TensorRT engine. Default: same folder/name "
            "as ONNX with .engine."
        ),
    )

    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the checkpoint to ONNX before inspecting the artefacts.",
    )

    parser.add_argument(
        "--build-engine",
        action="store_true",
        help="Build a TensorRT engine from ONNX using NVIDIA trtexec.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Request FP16 TensorRT engine building (used with --build-engine).",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for ONNX export (default: 640).",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Folder to scan for model artefacts (default: current folder).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    weights_path = args.weights.expanduser().resolve()

    onnx_path = (
        args.onnx.expanduser().resolve()
        if args.onnx is not None
        else weights_path.with_suffix(".onnx")
    )

    engine_path = (
        args.engine.expanduser().resolve()
        if args.engine is not None
        else onnx_path.with_suffix(".engine")
    )

    root = args.root.expanduser().resolve()

    # Ensure the default model output directory exists when exporting/building.
    if args.export_onnx or args.build_engine:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if args.build_engine:
        engine_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nVISIONPORT - MODEL ARTEFACT LIFECYCLE INSPECTOR")
    print("Project folder:", root)
    print("PyTorch checkpoint:", weights_path)

    print_file_inventory(root)
    report_gpu_status()
    inspect_checkpoint(weights_path)

    if args.export_onnx:
        exported_path = export_onnx(
            weights_path,
            onnx_path,
            args.imgsz
        )

        if exported_path is not None:
            onnx_path = exported_path

            if args.engine is None:
                engine_path = onnx_path.with_suffix(".engine")

    if args.build_engine:
        build_tensorrt_engine(
            onnx_path,
            engine_path,
            args.fp16
        )

    inspect_onnx(onnx_path)
    inspect_tensorrt_engine(engine_path)
    print_comparison()

    print("\n" + "=" * 72)
    print("INSPECTION COMPLETED")
    print("=" * 72)
    print(
        "The script has inspected the files and reported the available artefacts."
    )
    print(
        "For performance comparison, run a separate, controlled inference benchmark."
    )


if __name__ == "__main__":
    main()
