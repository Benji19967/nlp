from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    example = "My name is Wolfgang and I live in Berlin. I like Apple and my favourite fruit is the Apple."

    ner_results = nlp(example)
    print(ner_results)
