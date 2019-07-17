import logging
import os
import random

from clustering_data.get_raw_data import (
    get_f_names, parse_f_name_for_q_number,
    get_ddxs,
)
from src.standard_metrics import standard_metrics
from src.algorithms.connected_components import ConnectedComponentClusters
from src.evaluate_algorithm import AlgorithmEvaluator
from src.matching.get_matching_classifier import MatchingClassifier

logging.basicConfig(level=logging.DEBUG)


def compare_matchings(thresholds=(.4,.5,.6,.7,.8,.9,.95,.975), case_limit=30, solve_range=(5, 10)):
    normal_matching = ConnectedComponentClusters(MatchingClassifier().predict_proba)
    evaluator = AlgorithmEvaluator(
        solve_range=solve_range,
        nmi75 = standard_metrics["NMI_0.75"],
        nmi5=standard_metrics["Normalized Mutual Information"],
        ami=standard_metrics["Adjusted Mutual Information"]
    )
    evaluator.case_limit = case_limit
    results = {}
    for threshold in thresholds:
        logging.info("Testing at threshold %s" % threshold)
        normal_matching.threshold = threshold
        result = evaluator.evaluate_mean(normal_matching)
        logging.info("Normal matching: %s " % result)
        results["threshold %s" % threshold] = result
    return results


def save_matchings(thresholds=(.4, .5, .65, 0.8, .9), solve_range=(5, 10), case_limit=None):
    for f_name in get_f_names(case_limit):
        q_number = parse_f_name_for_q_number(f_name)
        normal_matching = ConnectedComponentClusters(MatchingClassifier().predict_proba)

        ddxs = get_ddxs(f_name)
        output_dir = "clustering_data/ccomp_clusters/clusters_q_%d" % q_number
        if solve_range:
            num_solves = random.randint(solve_range[0], solve_range[1])
            ddxs = ddxs[:num_solves]
            output_dir = "clustering_data/ccomp_clusters/clusters_q_%d_%d_solves" % (q_number, num_solves)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dxs = [dx for ddx in ddxs for dx in ddx]
        last_cluster_dict = None
        for threshold in thresholds:
            normal_matching.threshold = threshold
            output_file = os.path.join(output_dir, "clusters_at_%d.tsv" % (int(threshold*100)))
            cluster_dict = normal_matching.get_clustering_dict(dxs)
            if cluster_dict != last_cluster_dict:
                normal_matching.save_clusters_as_tsv(cluster_dict, output_file)
            last_cluster_dict = cluster_dict



"""
solve_range [5,10]
Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 6.4.0
Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46)
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
from src.connected_components import *
compare_matchings()
Testing at threshold 0.1
Normal matching: {'nmi': 0.39810648316500774}
Swapped matching: {'nmi': 0.013638538960784615}
Testing at threshold 0.2
Normal matching: {'nmi': 0.689282807604657}
Swapped matching: {'nmi': 0.03417799041178758}
Testing at threshold 0.4
Normal matching: {'nmi': 0.8886791999047952}
Swapped matching: {'nmi': 0.07922182949402214}
Testing at threshold 0.5
Normal matching: {'nmi': 0.9251752999106766}
Swapped matching: {'nmi': 0.1156297618289156}
Testing at threshold 0.6
Normal matching: {'nmi': 0.951430299966252}
Swapped matching: {'nmi': 0.19637093866957467}
Testing at threshold 0.7
Normal matching: {'nmi': 0.970935438377616}
Swapped matching: {'nmi': 0.43382129502819183}
Testing at threshold 0.8
Normal matching: {'nmi': 0.9811019093885405}
Swapped matching: {'nmi': 0.745824311935221}
Testing at threshold 0.9
Normal matching: {'nmi': 0.9868323488310019}
Swapped matching: {'nmi': 0.9747312579519833}
Testing at threshold 0.95
Normal matching: {'nmi': 0.9870807468415159}
Swapped matching: {'nmi': 0.9870347465723442}
Testing at threshold 0.975
Normal matching: {'nmi': 0.9856256773769845}
Swapped matching: {'nmi': 0.9860307691097399}
"""

"""
solve_range None
Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 6.4.0
Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
from src.connected_components import *
compare_matchings()
Testing at threshold 0.1
Normal matching: {'nmi': 0.11502185016701888} 
Swapped matching: {'nmi': 0.0075969728004324265} 
Testing at threshold 0.2
Normal matching: {'nmi': 0.37183104955107066} 
Swapped matching: {'nmi': 0.012340762376425054} 
Testing at threshold 0.4
Normal matching: {'nmi': 0.7624611812131832} 
Swapped matching: {'nmi': 0.04373825297816753} 
Testing at threshold 0.5
Normal matching: {'nmi': 0.8609109136813372} 
Swapped matching: {'nmi': 0.0598269758211953} 
Testing at threshold 0.6
Normal matching: {'nmi': 0.9123410555361833} 
Swapped matching: {'nmi': 0.10061205668773945} 
Testing at threshold 0.7
Normal matching: {'nmi': 0.9493852969410224} 
Swapped matching: {'nmi': 0.19002924870731674} 
Testing at threshold 0.8
Normal matching: {'nmi': 0.9710954122140384} 
Swapped matching: {'nmi': 0.5103515565708522} 
Testing at threshold 0.9
Normal matching: {'nmi': 0.9762929065630604} 
Swapped matching: {'nmi': 0.9534865169671033} 
Testing at threshold 0.95
Normal matching: {'nmi': 0.9762604615412147} 
Swapped matching: {'nmi': 0.9756534072259039} 
Testing at threshold 0.975
Normal matching: {'nmi': 0.9762604615412147} 
Swapped matching: {'nmi': 0.9762418124823677} 
"""