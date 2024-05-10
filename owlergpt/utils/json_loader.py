import cohere
import langchain.text_splitter
import os
import pandas as pd

from torch.utils.data import Dataset

class JSONDataset(Dataset):
    """
    A custom dataset class for loading entries from a JSONL file.
    """

    def __init__(self, path: str | os.PathLike, splitter: langchain.text_splitter, model_name: str,
                 chunk_size: int, chunk_overlap: int, chunk_prefix: str = None, record_type: str = 'query'):
        """
        Reads the JSONL file at the given path, if present.

        :param path: The path to the JSONL file.
        :param splitter: The text splitter to use for chunking texts.
        :param model_name: The model for which data should be loaded.
        :param chunk_size: The maximum number of tokens per chunk.
        :param chunk_overlap: The number of overlapping tokens between two chunks.
        :param chunk_prefix: An optional prefix to add to the start of each chunk, default: None.
        :param record_type: The type of record to be retrieved, i.e. query or document.
        """
        assert(record_type is not None)
        self.path = path
        self.jsonl = pd.read_json(open(path, 'r'), lines=True)
        self.splitter = splitter
        self.model = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_prefix = chunk_prefix
        self.record_type = record_type

    def __len__(self):
        """
        Returns the length of the JSONL dataset.

        :return: The dataset size.
        """
        return len(self.jsonl)

    def __getitem__(self, idx: int):
        """
        Retrieves the JSONL entry at the given index, splits it into smaller text chunks and returns them along with
        their ids and document number. If the text is not long enough or the text splitter was unable to create text
        chunks, None values are returned instead.

        :param idx: The index of the entry to retrieve.
        :return: The document name, ids and text chunks of the current entry.
        """
        doc_id = self.jsonl.iloc[idx]['_id']
        text = self.jsonl.iloc[idx]['text']
        # Check if there's enough text to process
        if not text or len(text.split()) < 5:
            return None, None, None # Skip documents with insufficient text

        # Split text into chunks
        if isinstance(self.splitter, cohere.Client):
            tokens = []
            token_response = self.splitter.tokenize(text=text, model=self.model)
            for token in token_response:
                if token[0] == "tokens":
                    tokens = tokens + token[1]
            text_chunks = _get_text_chunks(tokens, self.chunk_size, self.chunk_overlap, self.model, self.splitter)
        else:
            text_chunks = self.splitter.split_text(text)

        if not text_chunks:
            return None, None, None

        if self.chunk_prefix:
            text_chunks = [self.chunk_prefix + chunk for chunk in text_chunks]

        documents = [doc_id] * len(text_chunks)  # Use doc_id for all chunks
        documents = [str(doc) for doc in documents]
        ids = [f"{doc_id}-{i + 1}-{self.record_type}" for i in range(len(text_chunks))]  # Unique ID for each chunk

        return documents, ids, text_chunks

def collate_fn(data):
    """
    Custom function for returning data provided by the JSONL file as a batch. None values are removed and the lists of
    document names, ids and text chunks are flattened.

    :return: Flattened lists of document names, ids and chunks.
    """
    documents, ids, text_chunks = zip(*data)
    documents = sum(filter(None, documents), [])
    ids = sum(filter(None, ids), [])
    text_chunks = sum(filter(None, text_chunks), [])
    return documents, ids, text_chunks

def _get_text_chunks(tokens: list, chunk_size: int, chunk_overlap: int, model_name:str, client: cohere.Client):
    """
    Chunk the given list of tokens based on the chunk size and overlap and decode it using the Cohere tokenizer for the
    given model.

    :param tokens: A list of tokens representing an encoded document.
    :param chunk_size: The number of tokens in each chunk.
    :param chunk_overlap: The number of tokens that should overlap for two consecutive chunks.
    :param model_name: The name of the Cohere model which created the tokens.
    :param client: The Cohere client used to revert the tokens to text.
    :return: A list of decoded text chunks.
    """
    splits = []
    start_idx = 0
    cur_idx = min(start_idx + chunk_size, len(tokens))
    chunk_ids = tokens[start_idx:cur_idx]
    while start_idx < len(tokens):
        splits.append(client.detokenize(tokens=chunk_ids, model=model_name).text)
        if cur_idx == len(tokens):
            break
        start_idx += chunk_size - chunk_overlap
        cur_idx = min(start_idx + chunk_size, len(tokens))
        chunk_ids = tokens[start_idx:cur_idx]
    return splits
