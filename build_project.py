# build_project.py

from pathlib import Path

# --- Project Structure Definition ---
# This dictionary defines the complete directory structure for the project.
PROJECT_STRUCTURE = {
    "data": {
        "raw": [],
        "processed": [],
        "recordings": {
            "models": {"logs": []},
            "predictions": [],
            "evaluation_plots": [],
        },
        "sample": ["sample_gsr.csv"],  # Keep the sample file
    },
    "docs": [
        "proposal.tex",
        "appendix.tex",
        "information_sheet.tex",
        "consent_form.tex",
        "references.bib",
    ],
    "src": {
        "__init__.py": "",
        "capture": ["__init__.py", "video_capture.py", "gsr_capture.py"],
        "gui": ["__init__.py", "main_window.py"],
        "ml_models": ["__init__.py", "models.py"],
        "processing": [
            "__init__.py",
            "data_loader.py",
            "preprocessing.py",
            "feature_engineering.py",
        ],
        "evaluation": ["__init__.py", "visualization.py"],
        "scripts": [
            "__init__.py",
            "train_model.py",
            "evaluate_model.py",
            "inference.py",
        ],
        "utils": ["__init__.py", "data_logger.py"],
        "config.py": "",
        "main.py": "",
    },
    ".gitignore": "",
    "README.md": "",
    "requirements.txt": "",
}


def create_project_structure(base_path, structure):
    """
    Recursively creates directories and empty files based on a dictionary structure.
    """
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            # It's a directory
            print(f"Creating directory: {path}")
            path.mkdir(exist_ok=True)
            create_project_structure(path, content)
        elif isinstance(content, list):
            # It's a directory with specific files
            print(f"Creating directory: {path}")
            path.mkdir(exist_ok=True)
            for filename in content:
                file_path = path / filename
                print(f"  - Creating empty file: {file_path}")
                file_path.touch()
        else:
            # It's a single file
            print(f"Creating empty file: {path}")
            path.touch()


if __name__ == "__main__":
    project_root = Path(".")
    print(f"Setting up project structure in: {project_root.resolve()}")
    create_project_structure(project_root, PROJECT_STRUCTURE)
    print("\nProject structure created successfully.")
    print("Remember to populate the created Python files with the implemented code.")
    print("You should also add your .tex and .bib files to the 'docs' directory.")
