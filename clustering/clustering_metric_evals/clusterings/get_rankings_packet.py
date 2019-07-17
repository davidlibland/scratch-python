import glob
import os
from enum import Enum

RANKED_DIR = "clusterings/ranked"
RAW_DIR = "clusterings/raw"
OFFICIAL_CLUSTERING_FNAME = "no-dx_all-official.tsv"


def validate_rankings_packet(subdir_name):
    # Get the official clustering:
    official_clustering_file = os.path.join(RAW_DIR, subdir_name, OFFICIAL_CLUSTERING_FNAME)
    official_clustering = get_clustering(official_clustering_file)
    # Get the ranked clusterings:
    ranked_dir = os.path.join(RANKED_DIR, subdir_name)
    files = [f for f in glob.glob(ranked_dir + "/*.tsv", recursive=False)]
    files.sort()
    ranked_clusterings = [get_clustering(f_name) for f_name in files]
    matches = []
    for i, ranked in enumerate(ranked_clusterings):
        if ranked == official_clustering:
            matches.append(i)
    return "offical matches ranks %s for dir %s" % (matches, subdir_name)


def investigate_rankings_packet(subdir_name):
    # Get the original clustering:
    raw_dir = os.path.join(RAW_DIR, subdir_name)
    files = [f for f in glob.glob(raw_dir + "/*.tsv", recursive=False)]
    named_raw_clusterings = [(f_name, get_clustering(f_name)) for f_name in files]
    # Get the ranked clusterings:
    ranked_dir = os.path.join(RANKED_DIR, subdir_name)
    files = [f for f in glob.glob(ranked_dir + "/*.tsv", recursive=False)]
    files.sort()
    ranked_clusterings = [get_clustering(f_name) for f_name in files]
    results = {}
    for name, original in named_raw_clusterings:
        for i, ranked in enumerate(ranked_clusterings):
            if ranked == original:
                results[name] = i
    return "offical matches ranks %s for dir %s" % (matches, subdir_name)


def get_rankings_packet(subdir_name):
    # Get the official clustering:
    official_clustering_file = os.path.join(RAW_DIR, subdir_name, OFFICIAL_CLUSTERING_FNAME)
    official_clustering = get_clustering(official_clustering_file)
    # Get the ranked clusterings:
    ranked_dir = os.path.join(RANKED_DIR, subdir_name)
    files = [f for f in glob.glob(ranked_dir + "/*.tsv", recursive=False)]
    files.sort()
    ranked_clusterings = [get_clustering(f_name) for f_name in files]
    return {
        "official": official_clustering,
        "ranked": ranked_clusterings
    }


def get_clustering(f_name):
    label = ""
    data = {}
    with open(f_name, "r") as f:
        for l in f.readlines():
            if l[0] == "\t":
                data[l.strip()] = label
            else:
                label = l.strip()
    return data


class ClusterType(Enum):
    GMR_ONLY = "GMR"
    SMALL_ONLY = "SMALL"
    ALL = "ALL"


def get_clustering_packets(restriction: ClusterType):
    subdir_names = get_subdirs(restriction)
    for d in subdir_names:
        yield get_rankings_packet(d)


def get_subdirs(restriction: ClusterType):
    if restriction == ClusterType.GMR_ONLY:
        dir_regex = "/clustering_packet_gmr_q_*"
    elif restriction == ClusterType.SMALL_ONLY:
        dir_regex = "/clustering_packet_q_*"
    else:
        dir_regex = "/clustering_packet*"
    dir_names = [d for d in glob.glob(RANKED_DIR + dir_regex, recursive=False)]
    subdir_names = [os.path.split(d)[-1] for d in dir_names]
    return subdir_names


def get_clustering_lists(restriction: ClusterType):
    return map(rankings_packet_to_lists, get_clustering_packets(restriction))


def rankings_packet_to_lists(rankings_packet):
    items = sorted(rankings_packet["official"].keys())

    def dict_to_list(cluster_dict):
        return [cluster_dict[item] for item in items]

    return {
        "official": dict_to_list(rankings_packet["official"]),
        "ranked": [dict_to_list(d) for d in rankings_packet["ranked"]]
    }
