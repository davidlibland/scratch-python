import pickle as pkl
from functools import lru_cache
from itertools import islice

import numpy as np

from clustering_data.constants import TERM_EMBEDDINGS_MIMIC_50, TERM_EMBEDDING_PUBMED


@lru_cache()
def get_embeddings(*embedding_files):
    if len(embedding_files) == 0:
        embedding_files = [TERM_EMBEDDINGS_MIMIC_50, TERM_EMBEDDING_PUBMED]
    embedding_data = []
    total_embedding_dim = 0
    for embedding_file in embedding_files:
        with open(embedding_file, "rb") as fp:
            embedding_dict = pkl.load(fp)
        embedding_dim = len(list(islice(embedding_dict.values(), 1))[0])
        zero = np.zeros([embedding_dim])
        total_embedding_dim += embedding_dim
        embedding_data.append((embedding_dict, zero))

    def helper(wrd: str) -> np.ndarray:
        embeddings = [embedding_dict.get(wrd, zero)
                      for embedding_dict, zero in embedding_data]
        return np.concatenate(embeddings)

    helper.zero = np.zeros([total_embedding_dim])
    helper.f_names = embedding_files
    return helper
