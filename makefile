clean:
	rm -rf env

install:
	python3 -m venv env
	env/bin/pip install --upgrade pip
	env/bin/pip install $(shell cat requirements.txt | tr '\n' ' ')

mypy:
	env/bin/mypy
