import os
import click
import chromadb
import numpy as np
import pandas as pd
import torch

from chromadb.config import DEFAULT_TENANT
from flask import current_app
from owlergpt.utils import (choose_dataset_folder, filter_collections, calculate_metric, self_sim_score, nn_sim,
                            plot_results, get_embedding_indices)
from owlergpt.utils import AVAILABLE_METRICS, MATCH_DIM_METRICS, NEAREST_NEIGHBORS
from tqdm import tqdm

@current_app.cli.command("eval_ds")
def evaluate_ds_collections() -> None:
    """
    Compares embeddings created for a dataset using different embedding models with matching chunk size. The method
    offers different measures for evaluating similarity.
    """
    environ = os.environ
    default_chunk_size = int(environ.get("VECTOR_SEARCH_SENTENCE_DEFAULT_CHUNK_SIZE", 100))
    target_dimension = int(environ.get("EMBEDDING_DIMENSION"))  # Target dimension for the embeddings
    chunk_size = environ.get("CHUNK_SIZE")
    center = bool(int(environ.get("MEAN_CENTER")))
    batch_size = int(environ.get("BATCH_SIZE"))
    device = torch.device(environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"])
    nn_metric = environ.get("K_NN_METRIC")
    k = int(environ.get("K"))
    baseline = bool(int(environ.get("BASELINE")))

    if k < 3:
        raise ValueError("Number of retrieved results must be at least 3")

    # Init vector store
    chroma_client = chromadb.PersistentClient(
        path=environ["VECTOR_SEARCH_PATH"],
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    selected_folder = choose_dataset_folder(environ["DATASET_FOLDER_PATH"])

    if selected_folder is None:
        return

    try:
        db_name = f"{selected_folder}_{chunk_size}"

        if environ["VECTOR_SEARCH_TEXT_SPLITTER"]:
            name = environ["VECTOR_SEARCH_TEXT_SPLITTER"].split("/")[-1]
            db_name = f"{selected_folder}_{chunk_size}_{name}"

        chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=db_name)
    except ValueError:
        click.echo("No separate database found for dataset. Using default database.")

    # Fetch and list all collections
    collections = chroma_client.list_collections()
    if not collections:
        click.echo("No collections found.")
        return

    collections = [c.name for c in collections]
    collections.sort()

    # Ask the user for the evaluation metric to use
    metrics = AVAILABLE_METRICS
    metric = click.prompt(f"Please choose the evaluation metric {metrics}", type=str)
    match_dimension = metric in MATCH_DIM_METRICS

    # Check if the entered metric is valid
    if metric not in metrics:
        click.echo("Unsupported metric. Exiting.")
        return

    verb = "compare" if match_dimension else "retrieve"
    num_embeds = click.prompt(f"Please choose the number of embeddings to {verb} or 0 to {verb} all", type=int)
    num_queries = 1

    if num_embeds < 0:
        click.echo("Invalid number of embeddings. Exiting.")
        return

    if not match_dimension:
        num_queries = click.prompt(f"Please choose the number of queries to perform", type=int)

    if num_queries < 1:
        click.echo("Invalid number of queries. Exiting.")
        return

    collections_info, valid_collections = filter_collections(chroma_client, collections, target_dimension, chunk_size,
                                                             default_chunk_size, selected_folder, match_dimension)

    if not valid_collections:
        click.echo(f"No valid collections for chunk size {chunk_size} and dataset {selected_folder} found.")
        return
    else:
        click.echo(f"Found valid collections {valid_collections} for dataset {selected_folder} with chunk size "
                   f"{chunk_size}")

    mc = "_mc_" if center else "_"
    if baseline:
        mc += "base_"
    # Compare embeddings of the same elements in valid collections
    for dataset, collections in collections_info.items():
        if len(collections) > 1:
            click.echo(f"Comparing collections for dataset {dataset}...")

        results = {}
        lines = {}
        models = []
        sims_at_early_k = {}

        if metric in NEAREST_NEIGHBORS:
            for i in range(len(collections)):
                lines[i] = {}
            for i in range(k):
                results[i] = []
            for num in [3, 5, 10]:
                if k >= num:
                    sims_at_early_k[num] = {0: []}
        else:
            results[0] = []

        for i in range(len(collections)):
            model = collections[i]["embedding_model"]
            models.append(model)
            for j in tqdm(range(len(collections)), desc=f"Similarity for model {model}", total=len(collections)):
                if i == j and not baseline:
                    if metric in NEAREST_NEIGHBORS:
                        lines[i][j] = self_sim_score(metric)
                        for num in range(k):
                            results[num].append(self_sim_score(metric))
                        for key in sims_at_early_k:
                            sims_at_early_k[key][0].append(self_sim_score(metric))
                    else:
                        results[0].append(self_sim_score(metric))
                elif i > j and not baseline:
                    if metric in NEAREST_NEIGHBORS:
                        lines[i][j] = lines[j][i]
                        for num in range(k):
                            results[num].append(results[num][j * len(collections) + i])
                        for key in sims_at_early_k:
                            sims_at_early_k[key][0].append(sims_at_early_k[key][0][j * len(collections) + i])
                    else:
                        results[0].append(results[0][j * len(collections) + i])
                else:
                    collection1 = chroma_client.get_collection(name=collections[i]["name"])
                    collection2 = chroma_client.get_collection(name=collections[j]["name"])
                    queries1, queries2 = None, None
                    embeddings1 = collection1.get(include=["metadatas", "embeddings"])
                    embeddings2 = collection2.get(include=["metadatas", "embeddings"])
                    meta1 = embeddings1["metadatas"]
                    meta2 = embeddings2["metadatas"]
                    idx1, idx2, query_inds1, query_inds2 = get_embedding_indices(meta1, meta2, num_queries)

                    if not match_dimension:
                        queries1 = np.array(embeddings1["embeddings"])[query_inds1]
                        queries2 = np.array(embeddings2["embeddings"])[query_inds2]

                    embeddings1 = torch.tensor(embeddings1["embeddings"])[idx1]
                    embeddings2 = torch.tensor(embeddings2["embeddings"])[idx2]
                    sim, fig = calculate_metric(embeddings1, embeddings2, queries1, queries2, metric, batch_size,
                                                device, num_embeds, k, nn_metric, center, baseline)

                    if metric in NEAREST_NEIGHBORS:
                        lines[i][j] = sim
                        for num in range(k):
                            results[num].append(sim[num])
                            if num + 1 in {3, 5, 10}:
                                sims_at_early_k[num + 1][0].append(sim[num])
                    else:
                        results[0].append(sim)

                    if fig:
                        fig.show()
                        output_filename = os.path.join(
                            environ["VISUALIZATIONS_FOLDER_PATH"],
                            f"{dataset}_{chunk_size}_{collections[i]['embedding_model']}_vs_"
                            f"{collections[j]['embedding_model']}_{num_embeds}_{num_queries}{mc}{metric}.html")
                        fig.write_html(output_filename)

        if metric not in MATCH_DIM_METRICS:
            file_name = f"{dataset}_{chunk_size}_{num_embeds}{mc}{metric}"
            if metric in NEAREST_NEIGHBORS:
                plot_results(lines, file_name, environ["VISUALIZATIONS_FOLDER_PATH"], models, k)
                for key in sims_at_early_k:
                    file_name = f"{dataset}_{chunk_size}_{num_embeds}{mc}{metric}_top{key}"
                    plot_results(sims_at_early_k[key], file_name, environ["VISUALIZATIONS_FOLDER_PATH"], models, k)
            else:
                plot_results(results, file_name, environ["VISUALIZATIONS_FOLDER_PATH"], models, k)

        for i in results:
            j = i + 1
            df = pd.DataFrame(np.array(results[i]).reshape(len(collections), len(collections)), index=models,
                              columns=models)
            path = os.path.join(environ["EVAL_FOLDER_PATH"], f"{dataset}_{chunk_size}_{num_embeds}_{num_queries}_top{j}"
                                                             f"{mc}{metric}.csv")
            df.to_csv(path)
