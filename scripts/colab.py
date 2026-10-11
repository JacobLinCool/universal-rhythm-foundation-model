"""Drive Colab GPU runtimes for URFM experiments.

Datasets live on the Colab VM and on the Hugging Face Hub, never on this machine.
This script only sends source code, small scripts and the HF token to the VM, and
brings back small results.

Authentication uses Google Application Default Credentials (``gcloud auth
application-default login``). Session state is kept in ``.colab/sessions.json``,
which is git-ignored.

    uv run --extra colab python scripts/colab.py new             # L4 session "urfm-l4"
    uv run --extra colab python scripts/colab.py bootstrap       # code + deps + HF token
    uv run --extra colab python scripts/colab.py sync            # re-upload code after edits
    uv run --extra colab python scripts/colab.py exec job.py --timeout 3600
    uv run --extra colab python scripts/colab.py cli ls           # any other colab command

Stderr can contain authenticated proxy URLs, so URLs are always redacted.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".colab" / "sessions.json"
SESSION = os.environ.get("URFM_COLAB_SESSION", "urfm-l4")
REMOTE_ROOT = "/content/urfm"
HF_TOKEN = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
URL = re.compile(r"https?://[^\s\"']+")

REMOTE_SETUP = f"""
import os, subprocess, tarfile
os.makedirs("{REMOTE_ROOT}", exist_ok=True)
with tarfile.open("/content/urfm-src.tar.gz") as tar:
    tar.extractall("{REMOTE_ROOT}", filter="data")
subprocess.run(["pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "pip", "install", "--system", "-q", "-e", "{REMOTE_ROOT}[data]"], check=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
# a fresh interpreter sees the editable install; the running kernel does not
print(subprocess.run(["python", "-c", "import urfm; print('urfm', urfm.__version__)"],
                     capture_output=True, text=True, check=True).stdout.strip())
"""

REMOTE_TOKEN_DIR = """
import os; os.makedirs("/root/.cache/huggingface", mode=0o700, exist_ok=True)
"""

REMOTE_TOKEN_CHECK = """
import os
os.chmod("/root/.cache/huggingface/token", 0o600)
from huggingface_hub import whoami
print("huggingface:", whoami()["name"])
"""


def colab(*args: str, timeout: float | None = None) -> int:
    CONFIG.parent.mkdir(exist_ok=True)
    cmd = ["colab", "--auth", "adc", "--config", str(CONFIG), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    sys.stdout.write(URL.sub("[url]", result.stdout))
    if result.stderr:
        sys.stderr.write(URL.sub("[url]", result.stderr)[-5000:])
    return result.returncode


def exec_code(code: str, timeout: float = 600) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
    try:
        return colab("exec", "-s", SESSION, "-f", fh.name, "--timeout", str(timeout))
    finally:
        os.unlink(fh.name)


def source_archive(dest: Path) -> Path:
    files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    with tarfile.open(dest, "w:gz") as tar:
        for name in filter(None, files):
            path = ROOT / name.decode()
            if path.is_file() and not path.is_relative_to(CONFIG.parent):
                tar.add(path, arcname=name.decode())
    return dest


def cmd_new(args: argparse.Namespace) -> int:
    return colab("new", "-s", SESSION, "--gpu", args.gpu)


def cmd_sync(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        archive = source_archive(Path(tmp) / "urfm-src.tar.gz")
        code = colab("upload", "-s", SESSION, str(archive), "/content/urfm-src.tar.gz")
    return code or exec_code(REMOTE_SETUP, timeout=900)


def cmd_token(args: argparse.Namespace) -> int:
    if not HF_TOKEN.is_file():
        print(f"no Hugging Face token at {HF_TOKEN}; run `hf auth login` first", file=sys.stderr)
        return 1
    return (
        exec_code(REMOTE_TOKEN_DIR)
        or colab("upload", "-s", SESSION, str(HF_TOKEN), "/root/.cache/huggingface/token")
        or exec_code(REMOTE_TOKEN_CHECK)
    )


def cmd_bootstrap(args: argparse.Namespace) -> int:
    return cmd_sync(args) or cmd_token(args)


def cmd_exec(args: argparse.Namespace) -> int:
    return colab("exec", "-s", SESSION, "-f", args.file, "--timeout", str(args.timeout))


def cmd_cli(args: argparse.Namespace) -> int:
    return colab(*args.rest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("new", help="allocate a GPU session")
    p.add_argument("--gpu", default="L4")
    p.set_defaults(func=cmd_new)
    sub.add_parser("sync", help="upload the source tree and install it").set_defaults(func=cmd_sync)
    sub.add_parser("token", help="place the local HF token on the VM").set_defaults(func=cmd_token)
    sub.add_parser("bootstrap", help="sync + token").set_defaults(func=cmd_bootstrap)
    p = sub.add_parser("exec", help="run a local script on the VM")
    p.add_argument("file")
    p.add_argument("--timeout", type=float, default=600)
    p.set_defaults(func=cmd_exec)
    p = sub.add_parser("cli", help="pass arguments straight to the colab CLI")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_cli)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
