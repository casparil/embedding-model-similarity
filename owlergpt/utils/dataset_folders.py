""""
`utils.choose_dataset_folder` module: Choose a dataset folder from a list of available ones.
"""
import os
import click

def choose_dataset_folder(dataset_path: str):
    """
    Fetches and lists all available dataset folders to allow the user to choose for which dataset embeddings should be
    created or evaluated.

    :param dataset_path: The folder in which datasets are located.
    :return: The selected dataset folder.
    """
    dataset_folders = [f for f in os.listdir(dataset_path) if
                       os.path.isdir(os.path.join(dataset_path, f))]
    if not dataset_folders:
        click.echo("No dataset folders found.")
        return None

    click.echo("Available dataset folders:")
    for idx, folder_name in enumerate(dataset_folders, start=1):
        click.echo(f"{idx}. {folder_name}")

    prompt = "Please enter the number of the folder you want to process"

    folder_choice = click.prompt(prompt, type=int, default=1)
    if folder_choice < 1 or folder_choice > len(dataset_folders):
        click.echo("Invalid selection. Exiting.")
        return None

    selected_folder = dataset_folders[folder_choice - 1] if folder_choice > 0 else "all"
    click.echo(f"Selected folder: {selected_folder}")
    return selected_folder
