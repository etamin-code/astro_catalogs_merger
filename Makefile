PYTHON = python3.12
VENV_BIN = venv/bin
VENV_LINT_BIN = venv.lint/bin
PYTEST_OPTS = -n 8

.PHONY: update-requirements
update-requirements: packages.txt
	rm -rf venv
	virtualenv --python=$(PYTHON) venv
	$(VENV_BIN)/pip install -U pip
	$(VENV_BIN)/pip install -r $<
	$(VENV_BIN)/pip freeze > requirements.txt
	touch venv

venv: requirements.txt
	rm -rf $@
	virtualenv --python=$(PYTHON) $@
	$(VENV_BIN)/pip install -U pip
	$(VENV_BIN)/pip install -r $<

venv.lint: requirements.txt requirements.lint.txt
	rm -rf $@
	virtualenv --python=$(PYTHON) $@
	$(VENV_LINT_BIN)/pip install -U pip
	$(VENV_LINT_BIN)/pip install -r requirements.txt
	$(VENV_LINT_BIN)/pip install -r requirements.lint.txt

.PHONY: lint-black
lint-black: venv.lint
	$(VENV_LINT_BIN)/black --check --diff main.py

.PHONY: lint-ruff
lint-ruff: venv.lint
	$(VENV_LINT_BIN)/ruff check main.py

.PHONY: lint-mypy
lint-mypy: venv.lint
	$(VENV_LINT_BIN)/mypy --explicit-package-bases main.py

.PHONY: lint
lint: lint-black lint-ruff lint-mypy

.PHONY: fix-ruff
fix-ruff: venv.lint
	$(VENV_LINT_BIN)/ruff check --fix main.py

.PHONY: fix-black
fix-black: venv.lint
	$(VENV_LINT_BIN)/black main.py

.PHONY: fix-lint
fix-lint: fix-black fix-ruff

