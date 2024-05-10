""""
`utils.filter_collections` module: Filter collections to evaluate the constructed embeddings.
"""
import chromadb

def filter_collections(chroma_client: chromadb.PersistentClient, collections: list, target_dimension: int,
                       chunk_size: str, default_chunk_size: int, dataset_name: str,
                       match_dimension: bool = True) -> tuple[dict, list]:
    """
    Filters the given list of collection names present in the database according to the used chunk size, the dataset
    name, if present, and the target dimension, if the evaluation metric requires embeddings to be of the same size.

    :param chroma_client: The client used to access embeddings.
    :param collections: A list of collection names stored in the database.
    :param target_dimension: The target dimension the embeddings should have, only checked if match_dimensions is True.
    :param chunk_size: The chunk size used to create the embeddings to filter the collections.
    :param default_chunk_size: The default chunk size used to create embeddings.
    :param dataset_name: The name of the dataset whose embeddings should be compared.
    :param match_dimension: Whether the chosen evaluation metric requires matching embedding dimensions, default: True.
    :return: A dictionary containing information about valid collections along with a list of valid collection names.
    """
    # Parse collection names and filter by dataset and dimension
    collections_info = {}
    valid_collections = []  # To keep track of collections with the correct dimension

    for collection_name in collections:
        parts = collection_name.split("_")
        dataset = parts[0]

        if dataset_name != "all" and dataset != dataset_name:
            continue

        if len(parts) > 1:
            embedding_model = parts[1]
        else:
            embedding_model = "unknown_model"
        chunking_strategy = parts[2] if len(parts) > 2 else "default_strategy"
        chunking_size = parts[3] if len(parts) > 3 else default_chunk_size

        collection = chroma_client.get_collection(name=collection_name)
        sample_embedding = collection.get(include=["embeddings"], limit=1, offset=0)["embeddings"]
        if sample_embedding and chunking_size == chunk_size:
            if match_dimension and not len(sample_embedding[0]) == target_dimension:
                continue
            valid_collections.append(collection_name)

            if dataset not in collections_info:
                collections_info[dataset] = []

            collections_info[dataset].append({
                "name": collection_name,
                "embedding_model": embedding_model,
                "chunking_strategy": chunking_strategy,
                "chunking_size": chunking_size,
            })

    return collections_info, valid_collections
