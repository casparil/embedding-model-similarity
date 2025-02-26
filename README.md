# Beyond Benchmarks: Evaluating Embedding Model Similarity for Retrieval Augmented Generation Systems

This repository contains the code for our paper at the IR-RAG workshop at SIGIR 2024 "Beyond Benchmarks: Evaluating
Embedding Model Similarity for Retrieval Augmented Generation Systems".

[arXiv](https://arxiv.org/abs/2407.08275)

## Installation
- [Python 3.11+](https://python.org)
- [Python Poetry](https://python-poetry.org/)

Use the following commands to clone the project and install its dependencies:

```bash
git clone https://github.com/casparil/embedding-model-similarity
poetry env use 3.11
poetry install
```

---

## Configuring the application

Use environment variables to handle settings. Copy `.env.example` into a new `.env` file and edit it as needed.

```bash
cp .env.example .env
```

By default, `.env` is configured to use [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) as an embedding model (other models are available as
well, see the [Huggingface leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to include additional ones). Change the remaining properties as needed.
If you want to use a proprietary model, you will need to configure the corresponding API-key as well.

---

### Datasets
[List of datasets that can be used for evaluating RAG](https://github.com/beir-cellar/beir?tab=readme-ov-file)

---

## Ingesting Datasets

Place the dataset files under `./datasets` and run the following command to:
- Load the data in batches from the `corpus.jsonl` and `queries.jsonl` files.
- Generate text embeddings for the documents and questions. The text will be chunked automatically based on the
embedding model's tokenizer.
- Store these embeddings in a vector store, so they can be used for evaluation lateron.

```bash
poetry run flask ingest_ds # The dataset should contain corpus.jsonl, queries.jsonl

# Folder structure in datasets:
# -datasets/
# --fiqa/
# ---qrels/
# ---corpus.jsonl
# ---queries.jsonl
# --nfcorpus/
# ---qrels/
# ---corpus.jsonl
# ---queries.jsonl
# ....
```

---

## Evaluation

To calculate similarity scores for the stored embeddings generated by different models, run:

```bash
poetry run flask eval_ds
```
