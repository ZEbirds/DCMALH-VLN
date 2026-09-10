# DCMALH-VLN: Dual-Channel Memory Architecture for Zero-shot Long-Horizon VLN

**Final Year Project**

**Author:** Zibo Zheng (University of Nottingham Ningbo China)

This repository contains the official implementation of the dissertation: *Dual-Channel Memory Architecture for Zero-shot Long-Horizon VLN*.

## Abstract

This project proposes a Dual-Channel Memory Architecture combining a Hierarchical 3D Semantic Scene Graph and Vectorized Experience Memory for zero-shot long-horizon vision-language navigation. By maintaining both a topological understanding of the environment and a vectorized history of navigation experiences, our agent significantly reduces exploration redundancy and improves zero-shot planning capabilities in complex, unseen environments.

## Demonstrations

> Click any thumbnail below to watch the corresponding execution trajectory on YouTube.

| Trajectory Demo (Part 1) | Trajectory Demo (Part 2) |
| :---: | :---: |
| [![Demo 1](https://img.youtube.com/vi/QWZLfyjmAPs/mqdefault.jpg)](https://www.youtube.com/watch?v=QWZLfyjmAPs)<br>▶️ **[Case 01](https://www.youtube.com/watch?v=QWZLfyjmAPs)** | [![Demo 2](https://img.youtube.com/vi/5Kv-p34u5fI/mqdefault.jpg)](https://www.youtube.com/watch?v=5Kv-p34u5fI)<br>▶️ **[Case 02](https://www.youtube.com/watch?v=5Kv-p34u5fI)** |
| [![Demo 3](https://img.youtube.com/vi/HzB61G3deFE/mqdefault.jpg)](https://www.youtube.com/watch?v=HzB61G3deFE)<br>▶️ **[Case 03](https://www.youtube.com/watch?v=HzB61G3deFE)** | [![Demo 4](https://img.youtube.com/vi/hznD20m3eFA/mqdefault.jpg)](https://www.youtube.com/watch?v=hznD20m3eFA)<br>▶️ **[Case 04](https://www.youtube.com/watch?v=hznD20m3eFA)** |
| [![Demo 5](https://img.youtube.com/vi/va_C_lyXiMM/mqdefault.jpg)](https://www.youtube.com/watch?v=va_C_lyXiMM)<br>▶️ **[Case 05](https://www.youtube.com/watch?v=va_C_lyXiMM)** | [![Demo 6](https://img.youtube.com/vi/LCVSZ3UY6Rk/mqdefault.jpg)](https://www.youtube.com/watch?v=LCVSZ3UY6Rk) <br>▶️ **[Case 06](https://www.youtube.com/watch?v=LCVSZ3UY6Rk)** |
| [![Demo 7](https://img.youtube.com/vi/fyASYlE4weM/mqdefault.jpg)](https://www.youtube.com/watch?v=fyASYlE4weM)<br>▶️ **[Case 07](https://www.youtube.com/watch?v=fyASYlE4weM)** | [![Demo 8](https://img.youtube.com/vi/q-PdhDyapjg/mqdefault.jpg)](https://www.youtube.com/watch?v=q-PdhDyapjg)<br>▶️ **[Case 08](https://www.youtube.com/watch?v=q-PdhDyapjg)** |
| [![Demo 9](https://img.youtube.com/vi/UQvHFW2uC7Y/mqdefault.jpg)](https://www.youtube.com/watch?v=UQvHFW2uC7Y)<br>▶️ **[Case 09](https://www.youtube.com/watch?v=UQvHFW2uC7Y)** | [![Demo 10](https://img.youtube.com/vi/4PXpkbITPnQ/mqdefault.jpg)](https://www.youtube.com/watch?v=4PXpkbITPnQ)<br>▶️ **[Case 10](https://www.youtube.com/watch?v=4PXpkbITPnQ)** |

## Resources

* [Download Poster](https://github.com/ZEbirds/DCMALH-VLN/blob/main/poster.pptx)
* [Download Dissertation](https://github.com/ZEbirds/DCMALH-VLN/blob/main/thesis.pdf)


## Repository Structure

```text
/workspace/
├── cfg/                                    # Configuration files for ablation studies
│   ├── lhpr_vln_verify.yaml                # Full framework configuration (Ours)
│   ├── wo_frontier_score.yaml              # Ablation: w/o 4D Frontier Scoring
│   ├── wo_memory.yaml                      # Ablation: w/o Episodic Memory
│   ├── wo_safety_filter.yaml               # Ablation: w/o Safety Kinematic Filter
│   ├── wo_scene_graph.yaml                 # Ablation: w/o 3D Scene Graph
│   └── baseline.yaml                       # Baseline configuration
├── graph_eqa/                              # Core algorithmic engine
│   ├── planners/
│   │   ├── vlm_planner_lhpr_vln_gpt.py     # Semantic planner for Full Framework
│   │   └── Ablation_study_planner.py       # Specialized planner for ablation variants
│   ├── scene_graph/
│   │   └── scene_graph_sim.py              # Topological graph maintenance
│   ├── experience_memory.py                # Vectorized memory engine
│   └── ...                                 # Other utility modules
├── scripts/                                # Execution and analysis scripts
│   ├── viewer.py                           # Main runner for the Full Framework
│   ├── Ablation_study_runner.py            # Automated runner for ablation studies
│   ├── run_ablation.sh                     # Shell script for batch experiment management
│   ├── Evaluate_SPL&OSR.py                 # Quantitative metric calculation script
│   ├── draw_2d_topology.py                 # 2D hierarchical topology visualization
│   └── render_hierarchical_dsg.py          # 3D Scene Graph + Point cloud visualization
└── outputs/                                # Evaluation logs and data artifacts
    ├── lhpr_vln_verification_ours/         # Main benchmark evaluation (Full)
    ├── lhpr_vln_verification_baseline/     # Ablation study result
    ├── lhpr_vln_verification_wo_memory/    # Ablation study result
    ├── lhpr_vln_verification_wo_frontier_score/ # Ablation study result
    ├── lhpr_vln_verification_wo_Safety_Filter/  # Ablation study result
    ├── lhpr_vln_verification_wo_scene_graph/    # Ablation study result
    ├── lhpr_vln_verification_ab_ours/      # Ablation study result
    └── lhpr_vln_verification_old/          # Result in mid-term evaluation

```

## Dataset Preparation

This framework requires the HM3D scene dataset and the LHPR-VLN task dataset. Please ensure the datasets are placed in the correct directories as configured in the YAML files.

### 1. HM3D Scene Dataset

Download the HM3D dataset from the official repository: [matterport/habitat-matterport-3dresearch](https://www.google.com/search?q=https://github.com/matterport/habitat-matterport-3dresearch).
Extract and place the scene data under your local data directory. In the configuration file (`cfg/lhpr_vln_verify.yaml`), it should map to:
`scene_data_path: '/workspace/grapheqa_data_backup/scene_datasets/hm3d/train'`

*(Note: Habitat-Sim configuration is required to properly load HM3D environments. Ensure your Habitat-Sim setup matches standard GraphEQA requirements.)*

### 2. LHPR-VLN Task Dataset

Download the LHPR-VLN task dataset from HuggingFace: [Starry123/LHPR-VLN](https://www.google.com/search?q=https://huggingface.co/datasets/Starry123/LHPR-VLN).
Place the downloaded HuggingFace cache into your workspace. The default paths in the configuration are set as:

* `task_data: "/workspace/LH-VLN/LHPR-VLN/huggingface/hub/datasets--Starry123--LHPR-VLN/snapshots/af66785e0b456639088d673bc100dcc63f2df997/task"`
* `step_task_data: "/workspace/LH-VLN/LHPR-VLN/huggingface/hub/datasets--Starry123--LHPR-VLN/snapshots/af66785e0b456639088d673bc100dcc63f2df997/step_task"`

## Installation & Setup

We provide a Docker-based setup to ensure environmental consistency and reproducibility.

### 1. Docker Environment

You can directly pull the fully configured environment for this project:

```bash
docker pull zebirds/dcmalh-vln:v1

```

*(Alternatively, you can build upon the base image: `blakerbuchanan/grapheqa_for_habitat:0.0.1`)*

Run the container by mounting your local project directory and dataset backup:

```bash
docker run -it \
  --name dcmalh_vln_env \
  --gpus all \
  -v "/path/to/your/DCMALH-VLN:/workspace" \
  -v "/path/to/your/grapheqa_data_backup:/root/graph_eqa/data" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY \
  zebirds/dcmalh-vln:v1 \
  bash

```

*(Note: Replace `/path/to/your/...` with the absolute local paths on your machine).*

To restart and re-enter an existing container:

```bash
docker start dcmalh_vln_env
docker exec -it dcmalh_vln_env bash

```

### 2. Environment Variables

Once inside the container, configure the Python paths and the necessary API keys for the Vision-Language Model planner:

```bash
export PYTHONPATH="/workspace/graph_eqa:$PYTHONPATH"
export PYTHONPATH="/workspace:$PYTHONPATH"

export OPENAI_BASE_URL="your_base_url_here"
export OPENAI_API_KEY="your_api_key_here"

```

## Usage

### 1. Main Framework Evaluation

To run the full Dual-Channel Memory Architecture on the benchmark dataset:

```bash
cd /workspace
python scripts/viewer.py --cfg cfg/lhpr_vln_verify.yaml

```

### 2. Ablation Studies

To execute the ablation variants automatically across the dataset, use the provided shell script:

```bash
bash scripts/run_ablation.sh

```

Alternatively, to run a specific ablation variant manually:

```bash
python scripts/Ablation_study_runner.py --cfg cfg/wo_memory.yaml

```

### 3. Quantitative Metrics Calculation

After the experiments are completed, calculate the Success weighted by Path Length (SPL) and Oracle Success Rate (OSR):

```bash
python scripts/Evaluate_SPL&OSR.py --results_dir outputs/

```

### 4. Visualization

Generate visual representations of the topological maps and scene graphs from the saved trajectory logs:

```bash
# Render 2D hierarchical topology
python scripts/draw_2d_topology.py --data_path outputs/lhpr_vln_verification_ours/

# Render 3D Scene Graph with Point Cloud
python scripts/render_hierarchical_dsg.py --data_path outputs/lhpr_vln_verification_ours/

```
