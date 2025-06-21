#!/usr/bin/env python3
"""
Setup script for installing pre-commit hooks for the GSR-RGBT project.

This script installs pre-commit and sets up the hooks defined in .pre-commit-config.yaml.
It also installs the required dependencies for the hooks.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)


def check_pip():
    """Check if pip is installed."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print("Error: pip is not installed or not working properly.")
        sys.exit(1)


def install_pre_commit():
    """Install pre-commit if not already installed."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pre-commit"],
            check=True,
        )
        print("✅ pre-commit installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing pre-commit: {e}")
        sys.exit(1)


def install_dependencies():
    """Install dependencies required by the hooks."""
    dependencies = [
        "black",
        "isort",
        "flake8",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-comprehensions",
        "mypy",
        "types-requests",
        "types-PyYAML",
        "bandit",
        "pytest",
    ]
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + dependencies,
            check=True,
        )
        print("✅ Hook dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def install_hooks():
    """Install the pre-commit hooks."""
    try:
        subprocess.run(
            ["pre-commit", "install"],
            check=True,
        )
        print("✅ Pre-commit hooks installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing hooks: {e}")
        sys.exit(1)


def run_hooks_on_all_files():
    """Run the pre-commit hooks on all files."""
    print("\nRunning pre-commit hooks on all files (this may take a while)...")
    try:
        subprocess.run(
            ["pre-commit", "run", "--all-files"],
            check=False,  # Don't exit if hooks fail
        )
        print("\n✅ Pre-commit hooks have been run on all files.")
    except subprocess.CalledProcessError as e:
        print(f"Error running hooks: {e}")


def main():
    """Main function to set up pre-commit hooks."""
    print("Setting up pre-commit hooks for the GSR-RGBT project...\n")
    
    # Check prerequisites
    check_python_version()
    check_pip()
    
    # Check if .pre-commit-config.yaml exists
    if not Path(".pre-commit-config.yaml").exists():
        print("Error: .pre-commit-config.yaml not found in the current directory.")
        sys.exit(1)
    
    # Install pre-commit
    install_pre_commit()
    
    # Install dependencies
    install_dependencies()
    
    # Install hooks
    install_hooks()
    
    # Ask if user wants to run hooks on all files
    run_on_all = input("\nDo you want to run the hooks on all files now? (y/n): ").lower()
    if run_on_all == 'y':
        run_hooks_on_all_files()
    
    print("\n🎉 Setup complete! Pre-commit hooks will now run automatically on each commit.")
    print("You can also run them manually with 'pre-commit run' or 'pre-commit run --all-files'.")


if __name__ == "__main__":
    main()