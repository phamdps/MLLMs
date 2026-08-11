<div align="center">

# 🚦 Multimodal Large Language Models for Cross-Modal Spatiotemporal Transportation Prediction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg?style=flat-square)]()

*An advanced Multimodal Large Language Model (MLLM) framework designed for cross-modal spatiotemporal transportation forecasting and digital twin reasoning.*

</div>

---

## 📌 Core Architecture

<p align="center">
  <img src="assets/MLLMs.png" alt="MLLM Spatiotemporal Prediction Architecture" width="850"/>
  <br>
  <em><b>Figure 1:</b> Multimodal Large Language Models (MLLMs) paradigm shifting diverse input types into a unified semantic token space.</em>
</p>

### 🌐 Meso-Level Traffic Flow & Congestion Prediction

<p align="center">
  <img src="assets/FlowPrediction.png" alt="Spatiotemporal Flow Prediction Diagram" width="850"/>
  <br>
  <em><b>Figure 2:</b> Spatiotemporal flow estimation targeting road network operational states (flow, speed, and density).</em>
</p>

* **Multi-Scale Spatial Aggregation:** Bridges fine-grained **meso-level traffic flows** (sensor speeds/occupancy) with macro-level **travel demand** (zonal inflows/outflows) using explicit spatial mapping matrices ($A_{\text{meso} \rightarrow \text{macro}}$).
* **Cross-Modal Fusion:** Combines Spatial Graph Neural Networks (GNNs) with autoregressive Transformer backbones to align physical time-series and network topologies with natural language contexts.
* **Automatic Multi-Task Loss Balancing:** Implements **Kendall-Gal Homoscedastic Uncertainty Weighting** to automatically balance gradient magnitudes between flow and demand loss metrics.
* **Incident-Conditioned Forecasting:** Integrates unstructured text prompts (weather alerts, accident logs, POI descriptions) for non-recurrent congestion simulation.

---

## 🗂 Project Structure

```text
MLLMs project/
├── 📁 dataloader/                   # Data loading and prompt construction modules
│   ├── 📄 __init__.py               # Package initialization
│   ├── 📄 dataset.py                # Multimodal spatiotemporal dataloader
│   ├── 📄 drivelm_builder.py        # DriveLM dataset instance builder
│   ├── 📄 drivelm_prompt_builder.py # Natural language prompt generator for DriveLM
│   ├── 📄 load_real_data.py         # Real-world traffic dataset ingestion utilities
│   └── 📄 spatial_aggregation.py    # Meso-to-macro spatial mapping matrix (A_meso_macro)
├── 📁 evaluation/                   # Evaluation pipelines and loss functions
│   ├── 📄 __init__.py               # Package initialization
│   ├── 📄 mllm_evaluator.py         # MLLM generation and task evaluator
│   └── 📄 multi_task_loss.py        # Kendall-Gal uncertainty loss module
├── 📁 models/                       # Neural network backbones and encoders
│   ├── 📄 __init__.py               # Package initialization
│   ├── 📄 backbone.py               # MLLM fusion backbone engine
│   └── 📄 encoders.py               # Graph GNN & Time-series feature extractors
├── 📁 train/                        # Training execution logic
│   ├── 📄 __init__.py               # Package initialization
│   └── 📄 train.py                  # End-to-end multi-task training script
├── 📁 utils/                        # Shared utility functions and metrics
│   ├── 📄 __init__.py               # Package initialization
│   └── 📄 metrics.py                # Evaluation metrics (MAE, RMSE, MAPE)
├── 📄 __init__.py                   # Root package initialization
└── 📄 README.md                     # Project documentation

```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

Ensure you are using **Python 3.11**.

```bash
# Clone the repository
git clone [https://github.com/phamdps/MLLMs.git](https://github.com/phamdps/MLLMs.git)
cd MLLMs

# Create virtual environment using uv or conda
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


```

### 2. Configuration

Adjust model hyper-parameters, data paths, and task settings in your configuration files:

```yaml
data:
  meso_sensor_nodes: 500
  macro_zones: 25
  history_steps: 12   # Past 1 hour (5-min resolution)
  pred_steps: 12      # Future 1 hour

training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 50


```

### 3. Training

Launch the multi-task joint prediction training pipeline using the script in the `train` folder:

```bash
python train/train.py --config config/config.yaml


```

---

## 📊 Task Scenarios Supported

| Level | Task Scenario | Description |
| --- | --- | --- |
| **Micro** | **Trajectory Prediction** | Next $(x,y)$ coordinate and agent-agent interaction modeling. |
| **Meso** | **Traffic Flow Prediction** | Network-wide speed, occupancy, and link volume forecasting. |
| **Macro** | **Travel Demand Prediction** | Origin-Destination (OD) matrix and zone-level inflow/outflow prediction. |
| **Cross-Scale** | **Multi-Task Joint Prediction** | Simultaneous flow and demand prediction constrained by spatial physical consistency. |

---

## 🚀 Recent Experiment & Inference Results

> **Model in Action:** Multi-camera surveillance and autonomous driving feeds processed via **Qwen2-VL-7B-Instruct** (4-bit VRAM-optimized mode) generating structured spatial-temporal reasoning blocks.

```json
{
  "perception": {
    "current_traffic_density": "Low",
    "congestion_level": "Low",
    "bottlenecks": [],
    "stalled_vehicles": [],
    "safety_hazards": []
  },
  "prediction": {
    "potential_trajectories": [
      {
        "vehicle": "ego_vehicle",
        "trajectory": "straight",
        "speed": "30 mph"
      },
      {
        "vehicle": "other_vehicle",
        "trajectory": "accelerate",
        "speed": "40 mph"
      }
    ]
  },
  "planning": {
    "recommendation": "Maintain current speed and trajectory. No immediate adjustments needed."
  }
}


```
---

Here is an improved and expanded version of your reference section. It seamlessly integrates the newest iterations of your core models (like **Qwen2.5-VL** and updated local execution runners) alongside additional cutting-edge papers touching on multimodal traffic agents, edge deployment, and spatio-temporal foundation models.

---

## 📚 References & Reading List

* **Qwen2-VL & Qwen2.5-VL (Open Multimodal LLM):** Bai et al. (2024 / 2025), *[Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)* — Upgrades the vision-language series with native dynamic resolution processing, absolute time encoding for long-video event localization, and agentic computer/phone use.
* **TrafficGPT (Agentic Traffic Control):** Zhang et al. (2024), *[TrafficGPT: Viewing, Capturing, and Responding to Traffic Chaos with LLM](https://arxiv.org/abs/2305.09531)* — Demonstrates how LLM agents interface with traffic simulators to manage complex urban intersections.
* **Ollama & Local Execution Engines:** *[Ollama: Get up and running with Llama 3, Qwen 2, and other large language models locally](https://github.com/ollama/ollama)* — Framework enabling localized, privacy-compliant inference for edge-deployed digital twins.
* **LLMs in Transportation Management:** Zhao et al. (2026), *[Large Language Models in Transportation Systems Management and Operations: From Text Reasoning to Multi-modal Decision Support](https://arxiv.org/abs/2606.00991)* — Comprehensive survey evaluating how MM-LLMs integrate heterogeneous text, visual, and sensor inputs for operator-facing decision support.
* **Multimodal LLM for ITS:** Al-Tameemi et al. (2024), *[Multimodal LLM for Intelligent Transportation Systems](https://arxiv.org/abs/2412.11683)* — Proposes a unified 3D MLLM framework evaluating sequential, audio, and visual sensor telemetry for intelligent transportation.
* **Aurora (Multimodal TSFM):** Wu et al. (2025/2026), *[Aurora: Towards Universal Generative Multimodal Time Series Forecasting](https://arxiv.org/abs/2509.22295)* — Introduces modality-guided multi-head attention and prototype-guided flow matching for zero-shot time series synthesis.
* **HORAI (Frequency-Enhanced MFM):** Chen et al. (2026), *[Empowering Time Series Analysis with Large-Scale Multimodal Pretraining](https://arxiv.org/abs/2602.05646)* — Proposes a billion-scale multimodal time series corpus (MM-TS) leveraging endogenous images/text and exogenous news.
* **Earth Science Survey:** Zhao et al. (2026), *[Earth Science Foundation Models: From Perception to Reasoning and Discovery](https://arxiv.org/html/2605.12542v1)* — Comprehensive survey evaluating geospatial foundation models spanning perception, text reasoning, and agentic workflows.
* **GeoXplain Toolkit:** Koprolin et al. (2026), *[GeoXplain: On-the-Fly Visual Explanations for Weather Foundation Models](https://arxiv.org/abs/2607.05655)* — Interactive visual interpretation tool tailored for weather and climate foundation architectures like Microsoft Aurora.
* **Amazon Chronos:** Ansari et al. (2024), *[Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)* — Scaling tokenized scalar values into fixed vocabularies using language model architectures via cross-entropy loss.
* **ClimaX:** Nguyen et al. (2023), *[ClimaX: A foundation model for weather and climate](https://arxiv.org/abs/2301.10343)* — Flexible deep learning frameworks using custom tokenizers for geospatial grids.

---

## 🛡 License

Distributed under the **MIT License**. See `LICENSE` for more information.

