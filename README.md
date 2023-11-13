# Celeritas

Welcome to our branch, where we focus on Artifact Evaluation by providing an encapsulated implementation of the GraphSage model as an example. This README offers detailed instructions for installing and running Celeritas, along with an example application on two datasets: `ogbn-arxiv` and `ogbn-paper100M`.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Execution](#execution)


## Introduction

This repository is part of our effort to facilitate Artifact Evaluation in graph processing. We have encapsulated the GraphSage model as a demonstrative example of efficient node embedding in large graphs. While the paper presents a broader set of results, here we provide APIs and an implementation of the GraphSage model, allowing users to evaluate its performance on both medium and large-scale datasets.

## Installation

To install Celeritas and its dependencies, follow these steps:

1. **Initial Setup**:
   - Open your terminal and navigate to the root directory of this repository.
   - Run the following command:
     ```bash
     pip3 install .
     ```
   - This command checks your system environment, installs necessary software dependencies, compiles C/C++ code, and sets up Python bindings with pybind11.

## Execution

After installing Celeritas, you can run it using the following steps:

1. **Navigate to Script Directory**:
   - Change your current directory to `python_script` within the repository.

2. **Running the Model**:
   - To execute the script, use:
     ```bash
     python3 run.py
     ```
   - You can specify the dataset for the experiment using one of the following commands:
     - For `ogbn-arxiv` dataset:
       ```bash
       python3 run.py --experiment instance_arxiv
       ```
     - For `ogbn-paper100M` dataset:
       ```bash
       python3 run.py --experiment instance_paper100M
       ```

3. **Output Options**:
   - To display the results in the console, add the `--show_output_console` flag:
     ```bash
     python3 run.py --show_output_console
     ```
   - Without this flag, results are saved in the `python_script/results` directory.

