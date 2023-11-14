import argparse
import os
from pathlib import Path

from instance.run_obgn_arxiv import run_ogbn_arxiv
from instance.run_ogbn_paper100M import run_ogbn_paper100M


DEFAULT_DATASET_DIRECTORY = "datasets/"
DEFAULT_RESULTS_DIRECTORY = "results/"

if __name__ == "__main__":
    experiment_dict = {
        "instance_arxiv": run_ogbn_arxiv,
        "instance_papers100m": run_ogbn_paper100M,
    }

    parser = argparse.ArgumentParser(description='Reproduce experiments ')
    parser.add_argument('--experiment', metavar='experiment', type=str, choices=experiment_dict.keys(),
                        help='Experiment choices: %(choices)s')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='If true, the results of previously run experiments will be overwritten.')
    parser.add_argument('--enable_dstat', dest='enable_dstat', action='store_true',
                        help='If true, dstat resource utilization metrics.')
    parser.add_argument('--enable_nvidia_smi', dest='enable_nvidia_smi', action='store_true',
                        help='If true, nvidia-smi will collect gpu utilization metrics.')
    parser.add_argument('--dataset_dir', metavar='dataset_dir', type=str, default=DEFAULT_DATASET_DIRECTORY,
                        help='Directory containing preprocessed dataset(s). If a given dataset is not present'
                             ' then it will be downloaded and preprocessed in this directory')
    parser.add_argument('--results_dir', metavar='results_dir', type=str, default=DEFAULT_RESULTS_DIRECTORY,
                        help='Directory for output of results')
    parser.add_argument('--show_output', dest='show_output', action='store_true',
                        help='If true, the output of each run will be printed directly to the terminal.')
    parser.add_argument('--short', dest='short', action='store_true',
                        help='If true, a shortened version of the experiment(s) will be run')
    parser.add_argument('--num_runs', dest='num_runs', type=int, default=1,
                        help='Number of runs for each configuration. Used to average results.')

    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    args.results_dir = Path(args.results_dir)

    if not args.dataset_dir.exists():
        os.makedirs(args.dataset_dir)
    if not args.results_dir.exists():
        os.makedirs(args.results_dir)

    experiment_dict.get(args.experiment)(args.dataset_dir,
                                         args.results_dir,
                                         args.overwrite,
                                         args.enable_dstat,
                                         args.enable_nvidia_smi,
                                         args.show_output,
                                         args.short,
                                         args.num_runs)
