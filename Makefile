dev-install:
	pip install -r requirements_dev.txt
	pip install -e .

pre-commit:
	isort molclub tests
	black molclub tests
	flake8 molclub tests
	pylint molclub tests
	mypy molclub tests 

test:
	pytest -v

build:
	python -m build

coverage:
	coverage erase
	coverage run --include=tests/* -m pytest -ra
	coverage report -m