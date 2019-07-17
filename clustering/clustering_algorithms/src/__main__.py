

from pandas import DataFrame

from src.evaluate_algorithm import AlgorithmEvaluator
from src.standard_algorithms import base_algorithm_factories
from src.standard_metrics import base_metrics

OUTPUT_FILE_ROOT = "evaluations"


def run_evaluation(solve_range=(5, 10), case_limit=None, output_file_root=OUTPUT_FILE_ROOT):
    algorithm_factories = base_algorithm_factories
    evaluator = AlgorithmEvaluator(
        solve_range=solve_range,
        **base_metrics
    )
    metric_keys = list(base_metrics.keys())
    evaluator.case_limit = case_limit
    results = {}
    for alg_name, alg_factory in algorithm_factories.items():
        algorithm = alg_factory()
        alg_results = evaluator.evaluate_mean(algorithm)
        results[alg_name] = alg_results
    df = DataFrame.from_dict(
        orient='index',
        data={k: [v[m_k] for m_k in metric_keys] for k, v in results.items()},
        columns=metric_keys
    )
    if output_file_root is not None:
        if solve_range and case_limit:
            output_file = "%s_%d_cases_solves_%d-%d.tsv" % (
                output_file_root, case_limit, solve_range[0], solve_range[1]
            )
        elif solve_range:
            output_file = "%s_solves_%d-%d.tsv" % (
                output_file_root, solve_range[0], solve_range[1]
            )
        elif case_limit:
            output_file = "%s_%d_cases.tsv" % (
                output_file_root, case_limit
            )
        else:
            output_file = "%s.tsv" % output_file_root
        df.to_csv(output_file, sep="\t")
        print("File saved at %s" % output_file)
    return df
