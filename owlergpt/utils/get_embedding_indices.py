def _get_record_type_indices(meta1: list, meta2: list, inds1: list, inds2: list, num_queries: int):
    """
    Returns the indices of document and query embeddings for the two lists containing embedding metadata.

    :param meta1: A list of embedding metadata for the first model.
    :param meta2: A list of embedding metadata for the second model.
    :param inds1: A list of indices to check for the first model.
    :param inds2: A list of indices to check for the second model.
    :param num_queries: The maximum number of query indices to be returned.
    :return: A list of document and query indices for each model.
    """
    idx1, idx2, query_inds1, query_inds2 = [], [], [], []

    for idx in inds1:
        if meta1[idx]["record_type"] == "document":
            idx1.append(idx)
        elif len(query_inds1) < num_queries:
            query_inds1.append(idx)

    for idx in inds2:
        if meta2[idx]["record_type"] == "document":
            idx2.append(idx)
        elif len(query_inds2) < num_queries:
            query_inds2.append(idx)

    return idx1, idx2, query_inds1, query_inds2


def get_embedding_indices(meta1: list, meta2: list, num_queries: int):
    """
    Returns the indices of document and query embeddings given two lists of metadata containing additional information
    about embeddings generated for two different models. If the number of embeddings generated for the two models is
    different, the indices of matching embeddings ids are identified first.

    :param meta1: A list of embedding metadata for the first model.
    :param meta2: A list of embedding metadata for the second model.
    :param num_queries: The maximum number of query indices to be returned.
    :return: A list of document and query indices for each model.
    """
    records1 = [meta["record_id"] for meta in meta1]
    records2 = [meta["record_id"] for meta in meta2]

    records1_set = set(records1)
    records2_set = set(records2)

    common_records = records1_set.intersection(records2_set)

    match_idx1 = [i for i, record_id in enumerate(records1) if record_id in common_records]
    match_idx2 = [i for i, record_id in enumerate(records2) if record_id in common_records]

    return _get_record_type_indices(meta1, meta2, match_idx1, match_idx2, num_queries)
