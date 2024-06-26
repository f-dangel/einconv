.DEFAULT: help

help:
	@echo "install"
	@echo "        Install einconv and dependencies"
	@echo "uninstall"
	@echo "        Unstall einconv"
	@echo "lint"
	@echo "        Run all linting actions"
	@echo "install-dev"
	@echo "        Install all development tools"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "test-light"
	@echo "        Run pytest on the light part of test and report coverage"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "isort"
	@echo "        Run isort (sort imports) on the project"
	@echo "isort-check"
	@echo "        Check if isort (sort imports) would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "conda-env"
	@echo "        Create conda environment 'einconv' with dev setup"
	@echo "darglint-check"
	@echo "        Run darglint (docstring check) on the project"
	@echo "pydocstyle-check"
	@echo "        Run pydocstyle (docstring check) on the project"

.PHONY: install

install:
	@pip install -e .

.PHONY: uninstall

uninstall:
	@pip uninstall einconv

.PHONY: install-dev

install-dev:
	@pip install -e ."[test]"
	@pip install -e ."[lint]"

.PHONY: install-test

install-test:
	@pip install -e ."[test]"

.PHONY: test test-light

test:
	@pytest -vx --run-optional-tests=expensive --cov=einconv test

test-light:
	@pytest -vx --cov=einconv test

.PHONY: install-lint

install-lint:
	@pip install -e ."[lint]"

.PHONY: black black-check

black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

.PHONY: isort isort-check

isort:
	@isort .

isort-check:
	@isort . --check --diff

.PHONY: flake8

flake8:
	@flake8 .

.PHONY: darglint-check

darglint-check:
	@darglint --verbosity 2 einconv
	@darglint --verbosity 2 test

.PHONY: pydocstyle-check

pydocstyle-check:
	@pydocstyle --count .

.PHONY: conda-env

conda-env:
	@conda env create --file .conda_env.yml

.PHONY: lint

lint:
	make black-check
	make isort-check
	make flake8
	make darglint-check
	make pydocstyle-check
