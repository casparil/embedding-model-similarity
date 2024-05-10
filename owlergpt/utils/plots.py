import numpy as np
import os
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _save_heatmap(results: list, file_name: str, folder: str, models: list):
    """
    Creates a heatmap for the given scores of embedding similarity and saves them as a PDF.

    :param results: A list containing similarity scores for each pair of models.
    :param file_name: The name of the file that should be saved.
    :param folder: The folder in which to save the file.
    :param models: A list containing the names of all compared models.
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    df = pd.DataFrame(np.array(results).reshape(len(models), len(models)), index=models, columns=models)
    sns.heatmap(data=df, ax=ax, annot=True)
    plt.show()
    file_name += ".pdf"
    path = os.path.join(folder, file_name)
    with PdfPages(path) as pdf:
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')

def _save_cluster_heatmap(results: list, file_name: str, folder: str, models: list):
    """
    Creates a cluster heatmap for the given scores of embedding similarity and saves them as a PDF.

    :param results: A list containing similarity scores for each pair of models.
    :param file_name: The name of the file that should be saved.
    :param folder: The folder in which to save the file.
    :param models: A list containing the names of all compared models.
    """
    df = pd.DataFrame(np.array(results).reshape(len(models), len(models)), index=models, columns=models)
    clustermap = sns.clustermap(data=df, annot=True, fmt=".2f")
    file_name += "_cluster.pdf"
    path = os.path.join(folder, file_name)
    clustermap.figure.savefig(path, format='pdf', bbox_inches="tight")


def _save_lineplot(results: dict, file_name: str, folder: str, models: list, k: int):
    """
    Creates a lineplot with multiple lines using the data provided in the given dictionary.

    :param results: The data that should be plotted.
    :param file_name: The name of the file to be saved.
    :param folder: The folder where plots should be saved.
    :param models: A list of model names.
    :param k: The maximum x value.
    """
    assert(len(results) == len(models))
    x = [i + 1 for i in range(k)]
    for i in results:
        model = models[i]
        fig, ax = plt.subplots(layout='constrained', figsize=(20, 20))
        for j in results[i]:
            if i != j:
                ax.plot(x, results[i][j], label=f"{model}_vs_{models[j]}")
        fig.legend(loc='outside upper center', fontsize='xx-large', mode='expand', ncols=3)
        path = os.path.join(folder, f"{file_name}_{model}.pdf")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

def plot_results(results: dict, file_name: str, folder: str, models: list, k: int):
    """
    Visualizes the embedding similarity results passed in the given dictionary and stores the plots.

    :param results: The data to be visualized.
    :param file_name: The name of the file.
    :param folder: The folder where the plots should be stored.
    :param models: A list of model names for which similarities were calculated.
    :param k: Maximum x value for lineplots.
    """
    assert len(results) > 0
    if len(results) == 1:
        _save_heatmap(results[0], file_name, folder, models)
        _save_cluster_heatmap(results[0], file_name, folder, models)
    else:
        _save_lineplot(results, file_name, folder, models, k)
