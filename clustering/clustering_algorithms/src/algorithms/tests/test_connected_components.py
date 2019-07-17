from typing import Tuple, List

import numpy as np

from src.algorithms.connected_components import ConnectedComponentClusters
from src.algorithms.online_clustering_wrapper import OnlineClusteringWrapper
from src.online_clustering.connected_components import \
    FastConnectedComponentClusters
from src.online_clustering.tests.test_connected_components import \
    distance_linkage_score


def test_connected_components():
    known_matches = {
        ("stemi", "heart attack"),
        ("heart attack", "myocardial infarction"),
        ("pe", "pulmonary embolism")
    }
    def matcher(term_pairs: List[Tuple[str, str]]) -> List[float]:
        return [
            1 if (term_1, term_2) in known_matches or
                 (term_2, term_1) in known_matches
            else 0
            for (term_1, term_2) in term_pairs
        ]

    ccomp = ConnectedComponentClusters(matcher)

    ccomps = ccomp.txt_fit_predict([
        "stemi", "pe", "heart attack", "myocardial infarction",
        "pulmonary embolism", "nada"
    ])
    assert ccomps[0] == ccomps[2]
    assert ccomps[0] == ccomps[3]
    assert ccomps[1] == ccomps[4]
    assert ccomps[0] != ccomps[1]
    assert ccomps[0] != ccomps[5]
    assert ccomps[1] != ccomps[5]


def test_fast_connected_components():
    embedding = {
        "stemi": np.array([1.0]),
        "nstemi": np.array([1.1]),
        "stroke": np.array([-3.0]),
        "fracture": np.array([2.8]),
        "broken_bone": np.array([3.0]),
    }

    ccomp = OnlineClusteringWrapper(
        online_clusterer_factory=lambda: FastConnectedComponentClusters(
            match_scores=distance_linkage_score,
            score_threshold=0.5
        ),
        embedding=embedding.get
    )

    ccomps = ccomp.txt_fit_predict([
        "stemi", "broken_bone", "nstemi", "stroke", "fracture",
    ])
    print(ccomps)
    assert ccomps[0] == ccomps[2]
    assert ccomps[1] == ccomps[4]
    assert ccomps[0] != ccomps[1]
    assert ccomps[0] != ccomps[3]
    assert ccomps[1] != ccomps[3]

    ccomp = OnlineClusteringWrapper(
        online_clusterer_factory=lambda: FastConnectedComponentClusters(
            match_scores=distance_linkage_score,
            score_threshold=0.5
        ),
        embedding=embedding.get
    )
