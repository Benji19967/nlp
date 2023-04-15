# Named Entity Recognition (NER)

## Getting started

```bash
make install
make start
```

## Example request
```bash
curl -X POST 127.0.0.1:8000  \
-H "Content-Type: application/json" \
-d '{"text": "Apple is my favorite fruit. I like IBM computers"}'
```
