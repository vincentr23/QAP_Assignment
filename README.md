# QAP Assignment

## Getting Started

Install [uv][1] and set up the project with `uv sync`.

Use the `make_dataset()` function from `qap_assignment.dataset` to download
the data for tha kra30a and tai40a instances.
The .dat files will be stored in data/raw/.

## Developing

Store Jupyter notebooks in the `notebooks/` directory. Follow the [naming
convention][2] for notebooks used by Cookiecutter Data Science.

You should add a cell at the top of notebooks with the following:
```
%load_ext autoreload
%autoreload 2
```
This should make code from the `qap_assignment` module importable.

Before commiting, use `ruff` to format your Python code:
```sh
uvx ruff format
uvx ruff check --select I --fix
```

[1]: https://docs.astral.sh/uv/getting-started/installation/
[2]: https://cookiecutter-data-science.drivendata.org/using-the-template/
