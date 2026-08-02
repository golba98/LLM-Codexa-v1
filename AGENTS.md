# Repository Guidelines

## Project Structure & Module Organization

`src/` contains the language-model configuration and model (`config.py`,
`model.py`), tokenizer code, and the dataset pipeline under `src/data/`.
Command-line entry points live in `scripts/`: prepare data, train or inspect a
tokenizer, and build or inspect token data. Keep reusable logic in `src/` and
leave argument parsing in scripts. Tests are executable modules in `tests/`;
small deterministic inputs belong in `tests/fixtures/`. `configs/smoke.yaml`
is the checked-in configuration fixture. Local datasets, tokenized outputs,
logs, and checkpoints belong in their respective ignored `data/`, `logs/`, and
`checkpoints/` directories.

## Development & Test Commands

Use the repository virtual environment when available:

```bash
source .venv/bin/activate
python -m pytest
python tests/test_model.py
python scripts/prepare_dataset.py --help
python scripts/train_tokenizer.py --help
```

`pytest` runs the full regression suite; individual test files can be run
directly because they use plain `assert` statements. Use each script's
`--help` output before running a pipeline: inputs and output locations are
explicit CLI arguments, and generated data must not be committed.

## Coding Style & Naming Conventions

Write Python with four-space indentation, type annotations, and short
docstrings on public functions. Follow the existing standard-library-first
import grouping and keep lines readable rather than compressing logic. Use
`snake_case` for functions, variables, files, and CLI flags; use `PascalCase`
for classes and dataclasses; use `UPPER_SNAKE_CASE` for constants. Prefer
`pathlib.Path`, dataclasses, and explicit validation with informative errors.
No formatter or linter is configured, so match nearby code.

## Testing Guidelines

Name test files `test_*.py` and test functions `test_<behavior>`. Add focused
coverage for valid behavior, malformed input, and reproducibility when a
pipeline changes. Keep tests self-contained with `TemporaryDirectory`; do not
depend on local files in ignored data directories. Run `python -m pytest`
before opening a pull request.

## Commits & Pull Requests

Recent history uses imperative, scoped subjects such as `Implement reproducible
BPE tokenizer pipeline`. Keep commits short and focused. Pull requests should
state the behavioral change, list validation performed, and call out any
configuration, dataset format, or generated-artifact impact. Link relevant
issues when available; include console output or screenshots only when they
clarify a user-visible result.
