from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from fastapi import FastAPI

MODEL = "dslim/bert-base-NER"
PIPELINE = "ner"

EXAMPLE = "My name is Wolfgang and I live in Berlin."
"I like Apple and my favourite fruit is the Apple."

fast_api = FastAPI()


class Entity(BaseModel):
    word: str
    entity: str
    start: int
    end: int
    index: int
    score: float


def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL)
    return model


if __name__ == "__main__":
    model = get_model()
    nlp = pipeline(PIPELINE, model=model, tokenizer=tokenizer)

    ner_results = [Entity(**e) for e in nlp(EXAMPLE)]
    print(ner_results)
