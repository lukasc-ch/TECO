# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**TECO -- The Experienced Code Optimizer** is a Python project. The repository is in its earliest stages; no source code or build configuration exists yet.

## Expected Toolchain

Based on the `.gitignore`, the project is expected to use:

- **Testing**: pytest (`pytest`)
- **Linting/formatting**: Ruff (`ruff check .`, `ruff format .`)
- **Type checking**: mypy (`mypy .`)
- **Package management**: uv, poetry, or pdm (not yet decided)
- **Notebooks**: Marimo (`.gitignore` includes marimo artifacts)

Update this file once `pyproject.toml` and the project structure are established.
