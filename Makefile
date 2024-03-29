dev-install:
	pip install -r requirements_dev.txt
	pip install -e .

format_all:
	mypy molclub tests
	isort molclub tests
	black --line-length 79 molclub tests
	flake8  molclub tests

format:
	mypy molclub
	isort molclub
	black --line-length 79 molclub
	flake8 molclub

test:
	pytest -v tests

build:
	python -m build

coverage:
	coverage erase
	coverage run --include=tests/* -m pytest -ra
	coverage report -m
