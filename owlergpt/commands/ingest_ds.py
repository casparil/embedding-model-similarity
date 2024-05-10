import os
import click
import cohere

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, TokenTextSplitter
from chromadb import PersistentClient, Settings
from chromadb.api.client import AdminClient
from chromadb.config import DEFAULT_TENANT
from chromadb.db.base import UniqueConstraintError
from flask import current_app
from openai import OpenAI
from owlergpt.utils import JSONDataset, collate_fn, choose_dataset_folder
from torch.utils.data import DataLoader
from tqdm import tqdm


OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
COHERE_MODELS = ["embed-english-v3.0"]


@current_app.cli.command("ingest_ds")
def ingest_dataset() -> None:
    """
    Generates sentence embeddings and metadata for documents fetched from a dataset folder and saves them in a vector
    store. Allows the user to select which folder to process.
    """
    environ = os.environ

    normalize_embeddings = environ["VECTOR_SEARCH_NORMALIZE_EMBEDDINGS"] == "true"
    model_name = environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL"]
    chunk_overlap = int(environ["VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP"])
    embedding_model = None

    if model_name not in OPENAI_MODELS and model_name not in COHERE_MODELS:
        embedding_model = SentenceTransformer(
            model_name, device=environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"],
        )

    selected_folder = choose_dataset_folder(environ["DATASET_FOLDER_PATH"])

    if selected_folder is None:
        return

    # Ask the user for the tokens_per_chunk value
    tokens_per_chunk = click.prompt("Please enter the tokens per chunk value (128, 256, 512, 1024)", type=int)

    # Check if the entered value is valid
    if tokens_per_chunk not in [128, 256, 512, 1024]:
        click.echo("Invalid tokens per chunk value. Exiting.")
        return

    # Use the tokens_per_chunk value when initializing the text_splitter
    if model_name in OPENAI_MODELS:
        text_splitter = TokenTextSplitter(
            model_name=model_name,
            chunk_overlap=chunk_overlap,
            chunk_size=tokens_per_chunk
        )
        transformer_model = model_name
        client = OpenAI(api_key=environ["OPENAI_KEY"])
    elif model_name in COHERE_MODELS:
        client = cohere.Client(environ["COHERE_KEY"])
        text_splitter = client
        transformer_model = model_name
    else:
        text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name,
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,  # Use the user-provided value
        )
        transformer_model = model_name.split("/")[-1]

    # Initialize vector store and create a new collection
    chroma_client = PersistentClient(
        path=environ["VECTOR_SEARCH_PATH"],
        settings=Settings(anonymized_telemetry=False),
    )

    admin_client = AdminClient.from_system(chroma_client._system)
    db_name = f"{selected_folder}_{tokens_per_chunk}"

    try:
        admin_client.create_database(db_name)
        click.echo(f"Created dataset-specific DB {db_name} to store embeddings.")
    except UniqueConstraintError:
        click.echo(f"Dataset-specific DB {db_name} already exists. Using it to store embeddings")

    chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=db_name)

    # Include VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL in the collection name
    collection_name = f"{selected_folder}_{transformer_model}_CharacterSplitting_{tokens_per_chunk}"

    try:
        # Attempt to create a new collection with the selected folder name
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": os.environ["VECTOR_SEARCH_DISTANCE_FUNCTION"]},
        )
    except UniqueConstraintError:
        # If the collection already exists, delete it and create a new one
        click.echo(f"Collection {collection_name} already exists. Removing and creating a new one.")
        chroma_client.delete_collection(name=collection_name)
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": os.environ["VECTOR_SEARCH_DISTANCE_FUNCTION"]},
        )

    total_records = 0
    total_embeddings = 0
    batch_size = int(environ.get("BATCH_SIZE"))

    # Process the batch of documents
    for filename in ['corpus.jsonl', 'queries.jsonl']:
        if filename == "queries.jsonl":
            record_type = "query"
        else:
            record_type = "document"
        dataset = JSONDataset(os.path.join(environ["DATASET_FOLDER_PATH"], selected_folder, filename), text_splitter,
                              model_name, tokens_per_chunk, chunk_overlap, environ.get("VECTOR_SEARCH_CHUNK_PREFIX"),
                              record_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
        total_records += dataset.__len__()
        for documents, ids, text_chunks in tqdm(dataloader, desc='| Computing embeddings |', total=len(dataloader)):
            if len(documents) == 0 or len(ids) == 0 or len(text_chunks) == 0:
                continue
            # Generate embeddings for each chunk
            if model_name in OPENAI_MODELS:
                embeddings = []
                data = client.embeddings.create(input=text_chunks, model=model_name).data
                for entry in data:
                    embeddings.append(entry.embedding)
            elif model_name in COHERE_MODELS:
                embeddings = client.embed(texts=text_chunks, model=model_name, input_type="search_" + record_type,
                                          embedding_types=['float']).embeddings.float
            else:
                embeddings = embedding_model.encode(text_chunks, normalize_embeddings=normalize_embeddings).tolist()

            # Prepare metadata for each chunk
            metadatas = [
                {"record_id": ids[i], "record_text": text_chunks[i], "record_type": record_type}
                for i in range(len(text_chunks))
            ]

            total_embeddings += len(embeddings)

            # Store embeddings and metadata in the vector store
            chroma_collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    click.echo(f"Processed {total_records} documents, generated {total_embeddings} embeddings.")
