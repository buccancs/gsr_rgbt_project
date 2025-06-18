#!/usr/bin/env python3
"""
build_project.py  –  scaffold or rebuild gsr_rgbt_project (idempotent)

Usage
-----
python build_project.py          # normal scaffold/update
python build_project.py --reset  # wipe all files/dirs except this script, rebuild
"""

import argparse
import os
import shutil
import stat
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
THIS_SCRIPT = Path(__file__).resolve().name


def run(cmd, cwd=PROJECT_ROOT):
    subprocess.run(cmd, cwd=cwd, check=True, text=True)


def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        print(f"[+] dir   {p.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[=] dir   {p.relative_to(PROJECT_ROOT)} (exists)")


def ensure_file(p: Path, content: str):
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"[+] file  {p.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[=] file  {p.relative_to(PROJECT_ROOT)} (exists)")


def on_rm_error(func, path, exc_info):
    """Clear read‑only bit and retry removal (Windows)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument(
    "--reset",
    action="store_true",
    help="delete everything (except this script) and rebuild",
)
args = ap.parse_args()

# --------------------------------------------------------------------------- #
# Optional RESET                                                              #
# --------------------------------------------------------------------------- #
if args.reset:
    print("🗑   --reset: removing current contents …")
    for item in PROJECT_ROOT.iterdir():
        if item.name == THIS_SCRIPT:
            continue
        if item.is_dir():
            shutil.rmtree(item, onerror=on_rm_error)
        else:
            item.unlink(missing_ok=True)
    print("✅  workspace cleared.\n")

# --------------------------------------------------------------------------- #
# 1. Directory skeleton                                                       #
# --------------------------------------------------------------------------- #
DIRS = [
    "data/sample",
    "src/capture",
    "src/gui",
    "src/utils",
    "scripts",
    "third_party/physiokit",
    "third_party/neurokit2",
    "third_party/pyshimmer",
]
for d in DIRS:
    ensure_dir(PROJECT_ROOT / d)

# --------------------------------------------------------------------------- #
# 2. Placeholder code & aux files                                             #
# --------------------------------------------------------------------------- #
FILES = {
    "src/capture/video_capture.py": "# TODO: Implement capture logic\n",
    "src/gui/main_window.py": "# TODO: Implement GUI\n",
    "src/utils/data_logger.py": "# TODO: Implement data logger\n",
    "src/main.py": "# TODO: Implement entry point\n",
    "README.md": "# gsr_rgbt_project\n\nInitial scaffold.\n",
    ".gitignore": ".venv/\n__pycache__/\n*.pyc\n.idea/\n.vscode/\n"
    "data/recordings/\n",
    "requirements.txt": "\n".join(
        [
            "numpy",
            "pandas",
            "opencv-python-headless",
            "PyQt5",
            "pyqtgraph",
            "neurokit2",
            "pyshimmer @ git+https://github.com/dariyab1/pyshimmer.git",
        ]
    )
    + "\n",
    "data/sample/sample_rgb.mp4": "Placeholder RGB video.",
    "data/sample/sample_thermal.mp4": "Placeholder thermal video.",
    "data/sample/sample_gsr.csv": (
        "timestamp,value\n" "2025-01-01T00:00:00Z,0.1\n" "2025-01-01T00:00:01Z,0.2\n"
    ),
}
for rel, text in FILES.items():
    ensure_file(PROJECT_ROOT / rel, text)

# --------------------------------------------------------------------------- #
# 3. Git initialisation                                                       #
# --------------------------------------------------------------------------- #
if not (PROJECT_ROOT / ".git").exists():
    print("[*] git   init")
    run(["git", "init"])
else:
    print("[=] git   repo exists")


# Make sure 'main' branch exists even if no commits yet
def ensure_main_branch():
    # If HEAD ambiguous, create an orphan main
    try:
        current = subprocess.check_output(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        current = None
    if current != "main":
        print("[*] git   switch/create 'main' branch")
        try:
            run(["git", "checkout", "-B", "main"])
        except subprocess.CalledProcessError:
            # last resort orphan
            run(["git", "checkout", "--orphan", "main"])
            run(["git", "rm", "-rf", "--cached", "."], cwd=PROJECT_ROOT)


ensure_main_branch()

# --------------------------------------------------------------------------- #
# 4. Git submodules                                                           #
# --------------------------------------------------------------------------- #
SUBS = [
    ("https://github.com/PhysioLab/PhysioKit.git", "third_party/physiokit"),
    ("https://github.com/neuropsychology/NeuroKit2.git", "third_party/neurokit2"),
    ("https://github.com/dariyab1/pyshimmer.git", "third_party/pyshimmer"),
]
gitmodules_txt = (
    (PROJECT_ROOT / ".gitmodules").read_text()
    if (PROJECT_ROOT / ".gitmodules").exists()
    else ""
)

for url, path in SUBS:
    if url in gitmodules_txt:
        print(f"[=] submodule registered: {path}")
        continue
    print(f"[*] git   submodule add {url} {path}")
    try:
        run(["git", "submodule", "add", url, path])
    except subprocess.CalledProcessError as e:
        if e.returncode == 128:
            print(f"[!] submodule already exists on disk: {path}")
        else:
            raise

# --------------------------------------------------------------------------- #
# 5. Commit scaffold (if anything changed)                                    #
# --------------------------------------------------------------------------- #
run(["git", "add", "."], cwd=PROJECT_ROOT)
if (
    subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT).returncode
    != 0
):
    run(["git", "commit", "-m", "Initial scaffold"])
    print("[+] git   committed scaffold")
else:
    print("[=] git   nothing new to commit")

print("\n✅  Build complete.")
