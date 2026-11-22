# Contributing to irregts

Thank you for your interest in contributing to **irregts**! Here's how to get started.

## Setting Up

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/irregts.git
   cd irregts
   ```

2. Create a virtual environment and install in editable mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

## Reporting Issues

- Use the [GitHub Issues](https://github.com/arpanghosh2000/irregts/issues) page.
- Include a minimal reproducible example when reporting bugs.
- Describe expected vs actual behavior clearly.

## Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. Make your changes with clear, descriptive commit messages.
3. Ensure existing example notebooks still run correctly.
4. Open a pull request with a description of **what** you changed and **why**.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use NumPy-style docstrings.
- Include type safety guards and meaningful warnings for edge cases.

## Areas Where Contributions Are Welcome

- Additional correlation methods (e.g., ICCF, JAVELIN interface)
- GPU acceleration for large-survey datasets
- New example notebooks for different astrophysical objects
- Unit tests and CI pipeline
- Documentation improvements
