dev-install:
	pip install -r requirements_dev.txt
	pip install -e .

pre-commit:
	mypy molclub tests
	isort molclub tests
	black molclub tests
	flake8 molclub tests

test:
	pytest -v tests

build:
	python -m build

coverage:
	coverage erase
	coverage run --include=tests/* -m pytest -ra
	coverage report -m
