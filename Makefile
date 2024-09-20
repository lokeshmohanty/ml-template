lint:
	poetry run pylint --fail-under=7.0 --recursive=y .

test:
	PYTHONPATH=./src poetry run pytest ./tests

