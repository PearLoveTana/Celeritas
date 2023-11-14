from celeritas.utils.preprocessing.dataset.ogbn_papers100m import OGBNPapers100M
import utils.executor as e
import utils.report_result as r
from pathlib import Path

def run_ogbn_paper100M(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):

    dataset_name = "ogbn_papers100m"

    paper100M_config_path = Path("python_script/instance/configs_yaml/ogbn_paper100M/ogbn_paper100M.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = OGBNPapers100M(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess(num_partitions=8192, sequential_train_nodes=True)
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(num_runs):
        e.run_config(paper100M_config_path, results_dir / Path("ogbn_papers100m/celeritas_paper100M"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "celeritas")

    r.print_results_summary([results_dir / Path("ogbn_papers100m/celeritas_paper100M")])
