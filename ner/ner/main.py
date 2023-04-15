from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL = "dslim/bert-base-NER"
PIPELINE = "ner"

EXAMPLE = "My name is Wolfgang and I live in Berlin."
"I like Apple and my favourite fruit is the Apple."

app = FastAPI()


class InputText(BaseModel):
    text: str


class Entity(BaseModel):
    word: str
    entity: str
    start: int
    end: int
    index: int
    score: float


def find_entities(text: str) -> List[Entity]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL)
    nlp = pipeline(PIPELINE, model=model, tokenizer=tokenizer)
    ner_results = [Entity(**e) for e in nlp(text)]
    return ner_results


@app.post("/")
def ner(text: InputText) -> List[Entity]:
    entities = find_entities(text=text.text)
    print(entities)
    return entities
