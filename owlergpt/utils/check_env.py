import os


def check_env() -> bool:
    """
    Checks that required env variables are available.
    Throws if properties are missing or unusable.
    """
    environ = os.environ

    for prop in [
        "DATASET_FOLDER_PATH",
        "VISUALIZATIONS_FOLDER_PATH",
        "VECTOR_SEARCH_PATH",
        "VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL",
        "VECTOR_SEARCH_DISTANCE_FUNCTION",
        "VECTOR_SEARCH_NORMALIZE_EMBEDDINGS",
        "VECTOR_SEARCH_CHUNK_PREFIX",
        "VECTOR_SEARCH_QUERY_PREFIX",
        "VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE",
        "VECTOR_SEARCH_TEXT_SPLITTER",
        "VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP",
        "VECTOR_SEARCH_SEARCH_N_RESULTS",
        "BATCH_SIZE",
        "EMBEDDING_DIMENSION",
        "MEAN_CENTER",
        "K_NN_METRIC",
        "K",
        "BASELINE"
    ]:
        if prop not in environ:
            raise Exception(f"env var {prop} must be defined.")

    return True
