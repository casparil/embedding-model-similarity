import click
import numpy as np
import plotly.express as px
import random
import torch

from sklearn.neighbors import NearestNeighbors
from torch.nn.functional import cosine_similarity, pairwise_distance

COSINE = "cosine"
EUCLIDEAN = "euclidean"
JACCARD = "jaccard"
RANK = "rank"
CKA = "cka"
AVAILABLE_METRICS = [COSINE, EUCLIDEAN, JACCARD, RANK, CKA]
MATCH_DIM_METRICS = [COSINE, EUCLIDEAN]
NEAREST_NEIGHBORS = [JACCARD, RANK]

def _pairwise_similarity(embeds1: torch.Tensor, embeds2: torch.Tensor, metric: str, device: torch.device):
    """
    Calculates the pairwise cosine similarity or Euclidean distance between the given tensors.

    :param embeds1: The first tensors of dimension (N, D).
    :param embeds2: The second tensors of dimension (N, D).
    :param metric: The metric to be used.
    :param device: The device on which calculations are performed.
    :return: The calculated pairwise scores.
    """
    if metric == COSINE:
        return cosine_similarity(embeds1.to(device), embeds2.to(device))
    elif metric == EUCLIDEAN:
        return pairwise_distance(embeds1.to(device), embeds2.to(device))
    else:
        raise NotImplementedError(f"Provided unsupported metric {metric} for pairwise similarity!")

def _mean_pairwise_similarity(embeds1: torch.Tensor, embeds2: torch.Tensor, metric: str, batch_size: int,
                              device: torch.device):
    """
    Calculates the pairwise cosine similarity or Euclidean distance between batches of the given tensors and sums them
    up. In the end, the mean score is returned.

    :param embeds1: A tensor of dimension (N, D).
    :param embeds2: A tensor of dimension (N, D).
    :param metric: The metric to use for calculating the pairwise scores.
    :param batch_size: The batch size used to split the tensors.
    :param device: The device on which calculations should be performed.
    :return: The mean score over all batches.
    """
    assert embeds1.shape == embeds2.shape
    embeds1_batches = embeds1.split(batch_size)
    embeds2_batches = embeds2.split(batch_size)
    sum = 0
    sims = []
    for embed1, embed2 in zip(embeds1_batches, embeds2_batches):
        sim = _pairwise_similarity(embed1, embed2, metric, device)
        sims = sims + sim.detach().cpu().tolist()
        sum += torch.sum(sim).detach().cpu()
    text = 'Cosine Similarities' if metric is COSINE else 'Euclidean Distance'
    fig = px.histogram(x=sims, labels={'x': text}, title=f'{text} Distribution')
    return sum / len(embeds1), fig

def _jaccard_sim(indices1: np.ndarray, indices2: np.ndarray):
    """
    Calculates the Jaccard similarity between two 2D arrays by dividing the number of overlapping entries by the union
    of entries per row. The mean score over all rows is returned in the end.

    :param indices1: The first array of indices of shape (N, D).
    :param indices2: The second array of indices of shape (N, D).
    :return: The mean Jaccard similarity.
    """
    inds = np.concatenate((indices1, indices2), axis=1)
    len_union = np.array([len(np.unique(i)) for i in inds])
    len_intersection = np.array([len(set(i).intersection(set(j))) for i, j in zip(indices1, indices2)])
    return np.mean(len_intersection / len_union)

def _get_rank_sum(indices1: np.ndarray, indices2: np.ndarray):
    """
    Computes the sum term for rank similarity given the two 1D-arrays containing the indices of the k-nearest neighbors
    of two sets of activations.

    :param indices1: One row of indices calculated for the first set of activations.
    :param indices2: One row of indices calculated for the second set of activations.
    :return: The calculated rank sum.
    """
    aux = np.concatenate((indices1, indices2))
    aux_sort_indices = np.argsort(aux, kind='mergesort')
    aux = aux[aux_sort_indices]
    mask = aux[1:] == aux[:-1]
    ar1_indices = aux_sort_indices[:-1][mask] + 1
    ar2_indices = aux_sort_indices[1:][mask] - indices1.size + 1
    rank_sum = np.sum([2 / ((1 + abs(i - j)) * (i + j)) for i, j in zip(ar1_indices, ar2_indices)])
    return rank_sum

def _rank_sim(indices1: np.ndarray, indices2: np.ndarray):
    """
    Computes the rank similarity between two sets of indices. Rank similarities are calculated for each pair of rows and
    averaged.

    :param indices1: The first array of indices of shape (N, D).
    :param indices2: The second array of indices of shape (N, D).
    :return: The mean rank similarity.
    """
    rank_sums = [_get_rank_sum(i, j) for i, j in zip(indices1, indices2)]
    len_intersection = np.array([len(set(i).intersection(set(j))) for i, j in zip(indices1, indices2)])
    factors = []
    for idx, elem1 in enumerate(len_intersection):
        if elem1 > 0:
            factors.append(1 / sum([1 / (i + 1) for i in range(int(elem1))]))
        else:
            factors.append(0)
    res = np.array(factors) * np.array(rank_sums)
    return np.mean(res)

def nn_sim(indices1: np.ndarray, indices2: np.ndarray, metric: str):
    """
    Calculates the similarity of two sets of indices representing the index of nearest neighbors using Jaccard or rank
    similarity.

    :param indices1: The first array of indices of shape (N, D).
    :param indices2: The second array of indices of shape (N, D).
    :return: The average similarity.
    """
    if metric == JACCARD:
        return _jaccard_sim(indices1, indices2)
    else:
        return _rank_sim(indices1, indices2)

def _nearest_neighbors(embeds1: np.ndarray, embeds2: np.ndarray, queries1: np.ndarray, queries2: np.ndarray,
                       metric: str, k: int, nn_function: str, baseline: bool = False):
    """
    Calculates the nearest neighbors for two sets of queries and returns the k indices of the closest embeddings for
    each query. After obtaining the indices, their similarity is calculated.

    :param embeds1: The first set of embeddings of shape (N, D1).
    :param embeds2: The second set of embeddings of shape (N, D2).
    :param queries1: The first set of queries of shape (M, D1).
    :param queries2: The second set of queries of shape (M, D2).
    :param metric: The metric to use for comparing nearest neighbors.
    :param k: The number of nearest neighbors to retrieve.
    :param nn_function: The metric to use for finding nearest neighbors.
    :param baseline: Whether to compute a baseline score, default: False.
    :return: A list of similarity scores of length k.
    """
    neigh1 = NearestNeighbors(n_neighbors=k, metric=nn_function, algorithm="brute")
    neigh1.fit(embeds1)

    neigh2 = NearestNeighbors(n_neighbors=k, metric=nn_function, algorithm="brute")
    neigh2.fit(embeds2)

    indices1 = neigh1.kneighbors(queries1, n_neighbors=k, return_distance=False)
    indices2 = neigh2.kneighbors(queries2, n_neighbors=k, return_distance=False)

    if baseline:
        np.random.shuffle(indices1)

    sims = []

    for i in range(k):
        j = i + 1
        sims.append(nn_sim(indices1[:,:j], indices2[:,:j], metric))
    return sims, None

# Code taken from https://haydn.fgl.dev/posts/a-better-index-of-similarity/
def _cka(A: torch.Tensor, B: torch.Tensor):
    # Mean center each neuron
    A = A - torch.mean(A, dim=0, keepdim=True)
    B = B - torch.mean(B, dim=0, keepdim=True)

    dot_product_similarity = torch.linalg.norm(torch.matmul(A.t(), B)) ** 2

    normalization_x = torch.linalg.norm(torch.matmul(A.t(), A))
    normalization_y = torch.linalg.norm(torch.matmul(B.t(), B))

    cka = dot_product_similarity / (normalization_x * normalization_y)

    dot_product_similarity.detach()
    normalization_x.detach()
    normalization_y.detach()
    A.detach()
    B.detach()
    del dot_product_similarity, normalization_x, normalization_y, A, B
    return cka, None

def _calculate_embed_metric(embeds1: torch.Tensor, embeds2: torch.Tensor, queries1: np.ndarray, queries2: np.ndarray,
                            metric: str, batch_size: int, device: torch.device, k: int, nn_function: str,
                            baseline: bool = False):
    """
    Calculates the similarity between two embedding matrices using the given metric.

    :param embeds1: Document embedding tensor of shape (N, D1).
    :param embeds2: Document embedding tensor of shape (N, D1) if the metric requires matching dimensions or (N, D2).
    :param queries1: Embedding of queries of shape (N, D1).
    :param queries2: Embedding of queries of shape (N, D2).
    :param metric: The metric to be used, required to be one of AVAILABLE_METRICS.
    :param batch_size: The batch size used to calculate pairwise similarity.
    :param device: The device on which calculations should be performed.
    :param k: The number of nearest neighbors to retrieve for Jaccard or rank similarity.
    :param nn_function: The function for determining nearest neighbors.
    :param baseline: Whether to compute a baseline score, default: False.
    :return: The calculated similarity.
    """
    assert len(embeds1) == len(embeds2)

    if metric in MATCH_DIM_METRICS:
        return _mean_pairwise_similarity(embeds1, embeds2, metric, batch_size, device)
    elif metric in NEAREST_NEIGHBORS:
        return _nearest_neighbors(np.array(embeds1), np.array(embeds2), queries1, queries2, metric, k, nn_function,
                                  baseline)
    elif metric == CKA:
        return _cka(embeds1, embeds2)
    else:
        raise NotImplementedError(f"Provided unsupported metric {metric} for embedding similarity!")

def _sample_embeddings(embeds1: torch.Tensor, embeds2: torch.Tensor, min_size: int, num_embeds: int,
                       baseline: bool = False):
    """
    Returns a subset of embeddings if the desired number of embeddings to be compared is lower than the available ones.
    If a baseline score should be computed, the first set of embeddings is shuffled randomly if all embeddings are used
    or two different random subsets are returned.

    :param embeds1: Document embedding tensor of shape (N, D1).
    :param embeds2: Document embedding tensor of shape (N, D1) if the metric requires matching dimensions or (N, D2).
    :param min_size: The maximum number of embeddings that can be compared.
    :param num_embeds: The number of embeddings that should be compared.
    :param baseline: Whether a baseline score should be computed.
    """
    if baseline:
        indices = torch.randperm(embeds1.shape[0])
        embeds1 = embeds1[indices]
    if num_embeds > 0:
        if min_size < num_embeds:
            click.echo(f"Chosen number of embeddings is larger than number of available ones {min_size}. "
                       f"Using all available.")
        else:
            indices = random.sample(range(min_size), num_embeds)
            embeds1 = embeds1[indices]
            embeds2 = embeds2[indices]
    return embeds1, embeds2

def calculate_metric(embeds1: torch.Tensor, embeds2: torch.Tensor, queries1: np.ndarray, queries2: np.ndarray,
                     metric: str, batch_size: int, device: torch.device, num_embeds: int, k: int, nn_function: str,
                     center: bool = False, baseline: bool = False):
    """
    Calculates the similarity between document embeddings after pre-processing the embedding vectors.

    :param embeds1: Document embedding tensor of shape (N, D1).
    :param embeds2: Document embedding tensor of shape (N, D1) if the metric requires matching dimensions or (N, D2).
    :param queries1: Embedding of queries of shape (N, D1).
    :param queries2: Embedding of queries of shape (N, D2).
    :param metric: The metric to be used, required to be one of AVAILABLE_METRICS.
    :param batch_size: The batch size used to calculate pairwise similarity.
    :param device: The device on which calculations should be performed.
    :param num_embeds: The number of embeddings to be compared.
    :param k: The number of nearest neighbors to retrieve for Jaccard or rank similarity.
    :param nn_function: The function for determining nearest neighbors.
    :param center: Whether embeddings should be mean-centered, default: False.
    :param baseline: Whether to compute a baseline score, default: False.
    :return: The calculated similarity.
    """
    min_size = min(len(embeds1), len(embeds2))
    assert k <= min_size
    embeddings1 = embeds1[:min_size]
    embeddings2 = embeds2[:min_size]
    embeddings1, embeddings2 = _sample_embeddings(embeddings1, embeddings2, min_size, num_embeds, baseline)

    if center:
        embeddings1 = embeddings1 - torch.mean(embeddings1, axis=0, keepdim=True)
        embeddings2 = embeddings2 - torch.mean(embeddings2, axis=0, keepdim=True)

    return _calculate_embed_metric(embeddings1, embeddings2, queries1, queries2, metric, batch_size, device, k,
                                   nn_function, baseline)

def self_sim_score(metric: str):
    """
    Returns the similarity score between two identical vectors for the given metric.

    :param metric: The name of the metric.
    :return: The self-similarity score.
    """
    if metric == COSINE or metric == JACCARD or metric == RANK or metric == CKA:
        return 1
    elif metric == EUCLIDEAN:
        return 0
    else:
        raise NotImplementedError(f"Cannot return self-similarity score of unsupported metric {metric}!")
