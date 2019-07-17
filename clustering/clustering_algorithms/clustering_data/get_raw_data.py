import glob
import json
import random
import re
from typing import List, Mapping

q_number_re = re.compile(r".*q_packet_(?P<number>\d*)$")

def get_ddxs(f_name: str) -> List[List[str]]:
    with open(f_name, "r") as f:
        data = json.load(f)
    classifier = get_official_classifer_from_data(data)

    def recognized(ies):
        for ie in ies["insight_elements"]:
            if classifier(ie["title"]) is None:
                return False
        return True

    return [
        [ie["title"] for ie in ies["insight_elements"]]
        for ies in data["insight_objs"] if recognized(ies)
    ]


def get_official_clustering(f_name: str) -> Mapping[str, str]:
    with open(f_name, "r") as f:
        data = json.load(f)
    clustering = {}
    for element in data["official_collective"][0]["data"]["elements"]:
        label = element["label"]
        sublabels = [sub_el["label"] for sub_el in element["subelements"]]
        clustering[label] = sublabels
    return clustering


def get_official_classifier(f_name: str):
    with open(f_name, "r") as f:
        data = json.load(f)
    return get_official_classifer_from_data(data)


def get_official_classifer_from_data(data):
    lookup_dict = {}
    for element in data["official_collective"][0]["data"]["elements"]:
        label = element["label"]
        lookup_dict[lexical_key_generator(label)] = label
        for sublabel in [sub_el["label"] for sub_el in element["subelements"]]:
            lookup_dict[lexical_key_generator(sublabel)] = label
    return lambda s: lookup_dict.get(lexical_key_generator(s))


def get_f_names(case_limit=None) -> List[str]:
    """
    This returns a list of clustering packet file names, limited by case_limit.
    """
    f_names = list(glob.glob("clustering_data/q_packets/*"))

    if case_limit is None:
        return f_names
    else:
        random.shuffle(f_names)
        return f_names[:case_limit]

def parse_f_name_for_q_number(f_name) -> int:
    """Parses a q-packet file name for the case number."""
    return int(q_number_re.match(f_name)["number"])



###########
# Helpers #
###########

def lexical_key_generator(input_text: str) -> str:
    """
    Normalizes the input by making it lower case, it returns a string in all
    non-alphanumeric characters have been removed, and all letters made
    lowercase.

    This is accomplished in three steps, so that any one can be easily
    toggled. An internal function removes punctuation and condenses any
    resulting extended blocks of whitespace into a single space.

    Parameter:
        input_text (:class:`str`): The input string.

    Returns:
        (:class:`str`)
    """
    def filter_for_alphanumeric(intermediate_text):
        intermediate_text = re.sub(r'[^ \w]', '', intermediate_text)
        return re.sub('\s+', ' ', intermediate_text)

    # Makes lowercase
    input_text = input_text.lower()
    # Removes non-alphanumeric characters
    input_text = filter_for_alphanumeric(input_text)
    # Removes whitespace
    input_text = re.sub(r' ', '', input_text)

    return input_text


def output_terms(f_name="clustering_data/terms.txt"):
    terms = set()
    for f in glob.glob("clustering_data/q_packets/*"):
        ddxs = get_ddxs(f)
        for ddx in ddxs:
            for dx in ddx:
                terms.add(dx)
    with open(f_name, "w") as f:
        f.write("\n".join(sorted(terms)))
