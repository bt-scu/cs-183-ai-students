# CSCI 183 Final: The Effect of AI Usage on Student Performance
Authors: Benson Tran, Emily Ramos, Yann Chan

A data exploration, prediction, and visualization project studying how AI usage and other student features affect GPA.

## Prerequisites
- Python 3.12+
- uv
- Git

## Setup
```bash
git clone https://github.com/bt-scu/cs-183-ai-students.git
cd cs-183-ai-students
uv sync
```

## Run
```bash
uv run jupyter notebook
```

## Tech Stack
- **Language:** Python
- **Environment / Dependency Management:** uv
- **Models:** scikit-learn (linear regression), XGBoost, neural networks (PyTorch)
- **Interface:** Jupyter Notebook

## Dev Notes
Adding and removing Python packages:
```bash
uv add <package>
uv remove <package>
```

1. Always use `uv add`. Never use `pip install` or `!pip install` in a notebook. It won't be recorded, and teammates will get a `ModuleNotFoundError`.
2. Commit both `pyproject.toml` and `uv.lock`.
3. After pulling a teammate's changes, run `uv sync` to install anything new.


