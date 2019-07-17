from sklearn.cluster import (
    AffinityPropagation, AgglomerativeClustering,
    DBSCAN, Birch,
)

from clustering_data.constants import (
    TERM_EMBEDDING_PUBMED,
    TERM_EMBEDDINGS_MIMIC_50,
)
from clustering_data.get_embedding import get_embeddings
from src.algorithms.connected_components import ConnectedComponentClusters
from src.algorithms.online_clustering_wrapper import OnlineClusteringWrapper
from src.algorithms.sklearn_wrapper import SKLearnWrapper
from src.matching.get_matching_classifier import MatchingClassifier
from src.online_clustering.connected_components import \
    FastConnectedComponentClusters

standard_classifier = MatchingClassifier()

pairwise_matching = ConnectedComponentClusters(
    matcher=standard_classifier.predict_proba
)


def get_pairwise_matching_at_threshold(threshold):
    def lazy_matcher():
        pairwise_matching.threshold = threshold
        return pairwise_matching
    return lazy_matcher


def get_fast_pairwise_matching(threshold, num_neighbors=5):
    def matcher():
        return OnlineClusteringWrapper(
            online_clusterer_factory=lambda: FastConnectedComponentClusters(
                match_scores=standard_classifier.predict_proba_from_embeddings,
                score_threshold=threshold,
                num_neighbors=num_neighbors
            ),
            embedding=standard_classifier.embedding
        )
    return matcher



def get_pairwise_matching(threshold):
    def matcher():
        ccomp = ConnectedComponentClusters(
            matcher=standard_classifier.predict_proba
        )
        ccomp.threshold=threshold
        return ccomp
    return matcher

standard_embedding_data = [TERM_EMBEDDINGS_MIMIC_50, TERM_EMBEDDING_PUBMED]

standard_embedding_function = get_embeddings(*standard_embedding_data)


base_algorithm_factories = {
    **{
        "affinity_propagation": lambda: SKLearnWrapper(
            AffinityPropagation(),
            *standard_embedding_data
        ),
        "ward_hierarchical": lambda: SKLearnWrapper(
            AgglomerativeClustering(linkage="ward"),
            *standard_embedding_data
        ),
        "cosine_single_linkage_hierarchical": lambda: SKLearnWrapper(
            AgglomerativeClustering(linkage="single", affinity="cosine"),
            *standard_embedding_data
        ),
        "euclidean_single_linkage_hierarchical": lambda: SKLearnWrapper(
            AgglomerativeClustering(linkage="single"),
            *standard_embedding_data
        ),
        "dbscan_.5": lambda: SKLearnWrapper(
            DBSCAN(metric="cosine", eps=0.5),
            *standard_embedding_data
        ),
        "dbscan_1": lambda: SKLearnWrapper(
            DBSCAN(metric="cosine", eps=.75),
            *standard_embedding_data
        ),
        "birch_mimic_.5": lambda: SKLearnWrapper(
            Birch(threshold=.5),
            TERM_EMBEDDINGS_MIMIC_50
        ),
        "birch_mimic_.75": lambda: SKLearnWrapper(
            Birch(threshold=.75),
            TERM_EMBEDDINGS_MIMIC_50
        ),
        "birch_pubmed_.5": lambda: SKLearnWrapper(
            Birch(threshold=.5),
            TERM_EMBEDDING_PUBMED
        ),
        "birch_pubmed_.75": lambda: SKLearnWrapper(
            Birch(threshold=.75),
            TERM_EMBEDDING_PUBMED
        )
      },
    **{
        "fast_pairwise_matching_at_%.2f" % t: get_fast_pairwise_matching(t)
        for t in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
      },
    **{
        "pairwise_matching_at_%.2f" % t: get_pairwise_matching_at_threshold(t)
        for t in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
      },
}