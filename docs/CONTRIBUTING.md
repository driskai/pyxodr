# Guidelines for contributing

## Types of changes
Here you will find a guide for how to propose different types of changes to `pyxodr`.

### Bug fixes and small tweaks
Create a pull request with your changes. Include a description of the need for the change and what you have done in the PR. If the bug is non-trivial then consider adding a test for it.

### Larger problems or changes
If you want to propose a more significant change then create an issue with the label `enhancement` or `bug` as well as your pull request. Larger changes should always include tests.

### New features
Create an issue with the label `enhancement` and include a description of the feature.

### Other changes
For any other changes, questions or problems create an issue using the label you feel most appropriate. We will be happy to discuss it there!

## Testing
All tests should use `pytest`. In `tests/example_xodr_files.py` you will find some useful fixtures for testing on the ASAM example OpenDRIVE files.

## Style guidelines
The code style is `black`. The pre-commit hooks will ensure that all style requirements are met. These should be setup by installing the package with the `dev` extra and installing the hooks:
```
pip install "pyxodr[dev] @ git+https://github.com/driskai/pyxodr"
cd pyxodr
pre-commit install
```