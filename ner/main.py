from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from fastapi import FastAPI
from typing import List

MODEL = "dslim/bert-base-NER"
PIPELINE = "ner"

EXAMPLE = "My name is Wolfgang and I live in Berlin."
"I like Apple and my favourite fruit is the Apple."

app = FastAPI()


class Entity(BaseModel):
    word: str
    entity: str
    start: int
    end: int
    index: int
    score: float


@app.get("/")
def read_root() -> List[Entity]:
    return main()


def main() -> List[Entity]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL)
    nlp = pipeline(PIPELINE, model=model, tokenizer=tokenizer)

    ner_results = [Entity(**e) for e in nlp(EXAMPLE)]
    print(ner_results)
    return ner_results


if __name__ == "__main__":
    main()
