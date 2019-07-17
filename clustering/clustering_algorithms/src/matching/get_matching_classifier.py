from itertools import accumulate
from typing import Tuple, List
import pickle as pkl

import numpy as np

from clustering_data.constants import (
    TERM_EMBEDDINGS_MIMIC_50,
    TERM_EMBEDDING_PUBMED,
)
from clustering_data.get_embedding import get_embeddings


class MatchingClassifier:
    def __init__(self):
        self._clf_file = "src/matching/sym_tuned_pub_fresh_mimic_50_clf.pkl"
        self._mimic = get_embeddings(
            TERM_EMBEDDINGS_MIMIC_50
        )
        self._pubmed = get_embeddings(
            TERM_EMBEDDING_PUBMED
        )
        self._embedding_files = self._mimic.f_names + self._pubmed.f_names
        self._embeddings = [self._mimic, self._pubmed]
        self._embedding_sizes = [self._mimic.zero.size, self._pubmed.zero.size]
        with open(self._clf_file, "rb") as fp:
            clf_data = pkl.load(fp)
            self._clf_data = clf_data
            self._clf = clf_data["clf"]

    def predict(self, term_pairs: List[Tuple[str, str]]) -> List[float]:
        X = self.get_features_from_terms(term_pairs)
        return self._clf.predict(X)

    def predict_proba(self, term_pairs: List[Tuple[str, str]]) -> List[float]:
        X = self.get_features_from_terms(term_pairs)
        return list(self._clf.predict_proba(X)[:,1])

    def get_features_from_terms(self, term_pairs):
        vec_1_list = []
        vec_2_list = []
        for term_1, term_2 in term_pairs:
            vec_1_list.append(self.embedding(term_1))
            vec_2_list.append(self.embedding(term_2))
        vec_1 = np.vstack(vec_1_list)
        vec_2 = np.vstack(vec_2_list)
        return self.get_features_from_embeddings(vec_1, vec_2)

    def get_features_from_embeddings(self, vec_1, vec_2):
        X = get_partitioned_symmetric_feature_vec(
            vec_1=vec_1, vec_2=vec_2, partitions=self._embedding_sizes
        )
        return X

    def predict_proba_from_embeddings(
            self,
            vec_1: np.ndarray,
            vec_2: np.ndarray
    ) -> np.ndarray:
        X = self.get_features_from_embeddings(vec_1, vec_2)
        return self._clf.predict_proba(X)[:,1]

    def __repr__(self):
        parameters = {
            "type": "MatchingClassifier",
            "embeddings": self.embedding_files,
            "feature_type": "get_symmetric_feature_vec",
            "clf": self._clf_data
        }
        return str(parameters)

    @property
    def embedding_files(self) -> List[str]:
        return self._embedding_files

    def embedding(self, term: str):
        """Embeds a term."""
        return np.hstack([emb(term) for emb in self._embeddings])


def get_symmetric_feature_vec(embeddings, term_1, term_2):
    all_features = []
    for embedding in embeddings:
        vec1 = np.array(embedding(term_1))
        vec2 = np.array(embedding(term_2))
        norm1 = np.sqrt((vec1 ** 2).sum())
        norm2 = np.sqrt((vec2 ** 2).sum())
        dot = 0
        norm_dist_2 = 0
        if float(norm1) > 0 and float(norm2) > 0:
            dot = (vec1 * vec2).sum() / (norm1 * norm2)
            norm_dist_2 = ((vec1 / norm1 - vec2 / norm2) ** 2).sum()
        raw_dist_2 = ((vec1 - vec2) ** 2).sum()
        features = np.array([dot, raw_dist_2, norm_dist_2])
        all_features.append(features)
    return np.concatenate(all_features)


def get_partitioned_symmetric_feature_vec(
        vec_1: np.ndarray,
        vec_2: np.ndarray,
        partitions: List[int]) -> np.ndarray:
    """
    Partitions a vector and computes symmetric features along the partitions.
    """
    all_features = []
    splits = list(accumulate(partitions))[:-1]  # Drop the trailing one
    for part1, part2 in zip(np.split(vec_1, splits, axis=1), np.split(vec_2, splits, axis=1)):
        norm1 = np.sqrt((part1 ** 2).sum(axis=1, keepdims=True))
        norm2 = np.sqrt((part2 ** 2).sum(axis=1, keepdims=True))
        part1_nn = np.where(norm1 > 0, part1/norm1, part1)
        part2_nn = np.where(norm2 > 0, part2/norm2, part2)
        dot = (part1_nn * part2_nn).sum(axis=1, keepdims=True)
        norm_dist_2 = ((part1_nn - part2_nn) ** 2).sum(axis=1, keepdims=True)
        raw_dist_2 = ((part1 - part2) ** 2).sum(axis=1, keepdims=True)
        features = np.hstack([dot, raw_dist_2, norm_dist_2])
        all_features.append(features)
    return np.hstack(all_features)