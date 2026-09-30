"""Lazy backend imports and pinned source checks; no model is loaded on import."""

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PINS = {"flux": "ecee894ed2b1f3716d9d7326693061ec1a3105bb", "qwen": "fef717ffb01f407d2637584ed936c16db908587a"}
SOURCE_DIRS = {"flux": "ai-toolkit-flux", "qwen": "diffusers-qwen"}


def prepare_imports(backend: str) -> None:
    source = ROOT / "sources" / SOURCE_DIRS[backend]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_PINS[backend]:
        raise RuntimeError(f"Wrong {backend} source commit: {commit}")
    patch_name = "flux2_reference_branch_and_offload.patch" if backend == "flux" else "qwen21_local_pairs_comet.patch"
    actual_patch = subprocess.check_output(["git", "-C", str(source), "diff", "--binary"])
    if actual_patch != (ROOT / "patches" / patch_name).read_bytes():
        raise RuntimeError(f"{backend} checkout differs from the recorded source patch; reapply or record the intended patch")
    if backend == "flux":
        sys.path.insert(0, str(source))
        # Toolkit's broad package initializer imports unrelated audio/UI models.
        for name in ("extensions_built_in", "extensions_built_in.diffusion_models", "extensions_built_in.diffusion_models.flux2"):
            if name not in sys.modules:
                package = types.ModuleType(name)
                package.__path__ = [str(source / name.replace(".", "/"))]
                sys.modules[name] = package
        import sysconfig
        include = sysconfig.get_paths()["include"]
        if not (Path(include) / "Python.h").is_file():
            headers = sorted(Path.home().glob(".local/share/uv/python/cpython-3.11*/include/python3.11/Python.h"))
            if headers:
                include = str(headers[-1].parent)
        os.environ["CPATH"] = include + (":" + os.environ["CPATH"] if os.getenv("CPATH") else "")
    else:
        sys.path.insert(0, str(source / "src"))


def backend_module(config: dict):
    backend = config["model"]["backend"]
    prepare_imports(backend)
    return importlib.import_module(f"ba_dit.backends.{backend}_runtime")
