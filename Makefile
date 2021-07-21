init:
	poetry install

test:
	poetry run pytest

install_develop:
	poetry install

install:
	poetry install --no-dev

documentation:
	poetry run sphinx-apidoc -e -f DeepLearningBatchGeneratorUtils -o doc/
