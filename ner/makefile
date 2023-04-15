clean:
	rm -rf env

install:
	python3 -m venv env
	env/bin/pip install --upgrade pip
	env/bin/pip install $(shell cat requirements.txt | tr '\n' ' ')

image:
	docker build -t ner .

container:
	docker run --rm -it --entrypoint bash ner

mypy:
	env/bin/mypy 

start:
	uvicorn ner.main:app --reload
