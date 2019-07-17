import numpy as np

from src.matching.get_matching_classifier import MatchingClassifier


def test_matching_classifier():
    mc = MatchingClassifier()
    term_pairs = [("stemi", "nstemi"), ("stroke", "fracture")]
    matches = mc.predict(term_pairs)
    assert list(matches) == ["y", "n"], "Failed to match entities correctly."

    embedding_pairs = [
        (mc.embedding(t1), mc.embedding(t2)) for t1, t2 in term_pairs
    ]
    vec_1 = np.vstack([v1 for v1, v2 in embedding_pairs])
    vec_2 = np.vstack([v2 for v1, v2 in embedding_pairs])
    match_probs = mc.predict_proba_from_embeddings(vec_1, vec_2)
    assert match_probs[0] > 0.5, "%s and %s should match" % term_pairs[0]
    assert match_probs[1] < 0.5, "%s and %s should match" % term_pairs[1]
