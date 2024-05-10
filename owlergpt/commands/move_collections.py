import click
import chromadb
import os

from chromadb.api.client import AdminClient
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT
from chromadb.db.base import UniqueConstraintError
from flask import current_app
from owlergpt.utils import list_available_collections, choose_dataset_folder

@current_app.cli.command("move_col")
def move_collections() -> None:
    """
    Moves data persisted for a specific dataset from the default chromadb database to a new one.
    """
    environ = os.environ
    chunk_size = environ.get("CHUNK_SIZE")
    collections = list_available_collections()

    # Init vector store
    chroma_client = chromadb.PersistentClient(
        path=environ["VECTOR_SEARCH_PATH"],
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    selected_folder = choose_dataset_folder(environ["DATASET_FOLDER_PATH"])

    if selected_folder is None or not collections:
        return

    valid_collections = []

    for collection in collections:
        parts = collection.split("_")
        if parts[0] == selected_folder and parts[3] == chunk_size:
            valid_collections.append(collection)

    admin_client = AdminClient.from_system(chroma_client._system)

    try:
        db_name = selected_folder + "_" + chunk_size
        admin_client.create_database(db_name)
    except UniqueConstraintError:
        click.echo("Database already exists. Exiting.")
        return

    for collection_name in valid_collections:
        chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE)
        collection = chroma_client.get_collection(name=collection_name)
        data = collection.get(include=["metadatas", "embeddings", "documents"])
        chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=db_name)
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": os.environ["VECTOR_SEARCH_DISTANCE_FUNCTION"]},
        )
        chroma_collection.add(documents=data["documents"], embeddings=data["embeddings"], metadatas=data["metadatas"],
                              ids=data["ids"])
        click.echo("moved collection " + collection_name)
