# DeepReact: A workflow package for autonomous training of reactive deep learning potentials

DeepReact is a Python workflow manager designed to automate the
generation of reactive deep learning potentials (DLPs).

It connects classical molecular dynamics simulations, quantum chemical calculations,
dataset preparation, deep learning potential training, and active
learning into a reproducible workflow.

DeepReact does not implement molecular dynamics, quantum chemistry, or machine learning algorithms. Instead, it manages
existing computational tools, including:

-   LAMMPS for reactive molecular dynamics
-   mddatasetbuilder for extracting representative structures
-   Gaussian for quantum mechanical labeling
-   dpdata for dataset conversion
-   DeepMD-kit for DP model training

The goal of DeepReact is to simplify and standardize the construction of
reactive machine-learning potentials.

------------------------------------------------------------------------

# Workflow

The current workflow consists of training deep learning potentials and an optional active-learning loop.

## Core Pipeline

    Reactive MD (LAMMPS)
            │
            ▼
    Structure extraction (mddatasetbuilder)
            │
            ▼
    QC labeling (Gaussian)
            │
            ▼
    Quality check
            │
            ▼
    Dataset conversion (dpdata)
            │
            ▼
    Dataset splitting
            │
            ▼
    DP training (DeepMD-kit)
            │
            ▼
    Frozen model

## Active Learning  (optional)

After the initial model is trained, the active-learning loop refines it
by running additional MD with the DLP model, labelling new structures,
and retraining:

    DP-model MD (LAMMPS)
            │
            ▼
    Structure extraction (mddatasetbuilder)
            │
            ▼
    QC labeling (Gaussian)
            │
            ▼
    Quality check
            │
            ▼
    Dataset conversion (dpdata)
            │
            ▼
    Model evaluation
            │
            ▼
    Merge data + re-split + retrain (DeepMD-kit)
            │
            ▼
    Improved model

------------------------------------------------------------------------

# Installation

Clone the repository and install:

``` bash
pip install -e .
```

Requirements:

-   Python \>= 3.10
-   PyYAML

External programs should be installed separately:
-   LAMMPS, for reactive molecular dynamics
-   mddatasetbuilder, for extracting representative structures
-   Gaussian, for quantum mechanical labeling
-   dpdata, for dataset conversion
-   DeepMD-kit, for DP model training

------------------------------------------------------------------------

# Quick Start

Run a workflow with:

``` bash
deepreact run config.yaml
```
In this test version, run a workflow with:

``` bash
PYTHONPATH=. python -m deepreact.main run example/config.yaml
```

------------------------------------------------------------------------

# Configuration Reference

A full configuration file is shown below.

``` yaml
project:
    name: methane
    workdir: ./project

lammps:
    executable: lmp_mpi 
    input: ./example.lmp
    output: ./example.lammpstrj
    bonds: ./example.bonds.out

mddatasetbuilder:
    trajectory: ./example.lammpstrj
    bonds: ./example.bonds.out
    output: ./gaussian
    atoms: "C H O"
    name: data
    cutoff: 2.0
    stride: 1000
    nprocjob: 64
    keywords: "force b3lyp/3-21G* Geom=PrintInputOrient"
    gjf_inject: "%mem=128GB"   

gaussian:
    mode: export # mode: export = pause for manual execution; run = execute g16 automatically
    executable: g16
    input_dir: ./gaussian
    output_dir: ./gaussian/log

split:
    train_ratio: 0.7
    val_ratio: 0.2
    seed: 42

scripts:
    check: "" # empty = use built-in
    dpdata: ""
    split: ""

deepmd:
    executable: dp
    train: train.json

active_learning:
    enabled: true               # set to false to skip
    iterations: 1               # number of AL loops, this test version is recommended to be set to 1
    lammps_input: ./example_dp.lmp
    trajectory_output: ./example_dp.lammpstrj
    dptest_model: graph.pb
    dptest_system: ./deepmd_data_al
    dptest_output: ./dptest_out
    energy_rmse_threshold: 0.01 
    force_rmse_threshold: 0.1 
    train_json: ./train.json
    checkpoint: ./model.ckpt
```

The `active_learning` section is optional.  When enabled, the loop runs
*N* iterations (controlled by `iterations`) after the initial model is
trained. Each iteration produces a uniquely named model: `graph_al1.pb`,
`graph_al2.pb`, etc. 

------------------------------------------------------------------------

# Workflow Control

### Resume

If a calculation is interrupted:

``` bash
deepreact run config.yaml
```

DeepReact automatically skips completed stages and continues from the
last unfinished step.

### Manual Calculation

Set `gaussian.mode` to `export`.  DeepReact generates Gaussian input
files and pauses.  Run Gaussian manually, place the `.log` files in
the configured `output_dir`, then re-run the workflow.

### Active Learning

Active-learning iterations use a separate checkpoint directory
(`<workdir>/al_checkpoints/`).  Completed iterations are skipped on
re-run.

------------------------------------------------------------------------

# Citation

If you use DeepReact in your research, please cite the GitHub repository. An article introducing this software will be published later. Once the article is officially available, please cite it instead.

Alternatively, if you need to cite something right now, you are welcome to cite our article that shares the same methodology as DeepReact:

``` bash
Bin Chen, Kexin Chen, Yujie Zeng, Yuxuan Zhang, Mapping the synergistic co-pyrolysis reaction landscape of biomass and oil shale via deep learning potentials, Chemical Engineering Science (2026), doi: https://doi.org/10.1016/j.ces.2026.124771
```
