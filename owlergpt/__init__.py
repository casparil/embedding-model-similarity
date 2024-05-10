import os

from dotenv import load_dotenv
from flask import Flask

from owlergpt import utils


load_dotenv()


def create_app():
    """
    App factory (https://flask.palletsprojects.com/en/2.3.x/patterns/appfactories/)
    """
    app = Flask(__name__)

    # Note: Every module in this app assumes the app context is available and initialized.
    with app.app_context():
        utils.check_env()

        os.makedirs(os.environ["DATASET_FOLDER_PATH"], exist_ok=True)
        os.makedirs(os.environ["VISUALIZATIONS_FOLDER_PATH"], exist_ok=True)
        os.makedirs(os.environ["VECTOR_SEARCH_PATH"], exist_ok=True)
        os.makedirs(os.environ["EVAL_FOLDER_PATH"], exist_ok=True)

        from owlergpt import commands

        return app
