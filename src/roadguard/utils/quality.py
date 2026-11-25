"""Run the repository's independent checks and archive a review notebook."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from roadguard.utils.paths import write_text


def run_quality_gates(
    root: Path, run: Path, notebook_path: Path, *, notebook_env: dict | None = None
) -> dict:
    """Archive check logs and the executed copy of an artifact-only notebook."""
    env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    gates = {}
    commands = {
        "pytest": [shutil.which("python") or sys.executable, "-B", "-m", "pytest"],
        "ruff_lint": ["ruff", "check", "--no-cache", "."],
        "ruff_format": ["ruff", "format", "--check", "--no-cache", "."],
    }
    for name, command in commands.items():
        with (run / f"{name}.log").open("w", encoding="utf-8") as output:
            try:
                completed = subprocess.run(
                    command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, check=False
                )
                gates[name] = completed.returncode == 0
            except OSError as error:
                output.write(str(error))
                gates[name] = False
    try:
        import nbformat
        from nbclient import NotebookClient

        with TemporaryDirectory(prefix="roadguard-notebook-") as temp:
            notebook = nbformat.read(root / notebook_path, as_version=4)
            nbformat.validate(notebook)
            NotebookClient(notebook, kernel_name="python3", timeout=180).execute(
                cwd=str(root),
                env=env | (notebook_env or {}) | {"IPYTHONDIR": temp, "JUPYTER_RUNTIME_DIR": temp},
            )
            write_text(run / notebook_path.name, nbformat.writes(notebook))
    except Exception as error:
        write_text(run / "notebook_execution.log", f"{type(error).__name__}: {error}\n")
        gates["notebook"] = False
    else:
        write_text(run / "notebook_execution.log", f"Executed {notebook_path} successfully.\n")
        gates["notebook"] = True
    return gates
