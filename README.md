# TEAM & TEAM-LM – Transformer Earthquake Alerting Model (Modified for TensorFlow 2.15)

## Overview
This repository contains a **modified version** of the [Transformer Earthquake Alerting Model (TEAM)](https://doi.org/10.1093/gji/ggaa609) and its extension **TEAM-LM** for real-time earthquake early warning research.  

TEAM is a deep learning model designed for:
- Real-time estimation of **peak ground acceleration (PGA)**.  
- Real-time estimation of **earthquake magnitude** and **location** (TEAM-LM).  

The original implementation was built on TensorFlow 1.x.  
👉 **This version updates the implementation to run with TensorFlow 2.15**, ensuring compatibility with modern Python environments and GPU support.  

⚠️ **Note:** This repository is intended for **research purposes only**. It is not optimized for production or operational EEW systems.

---

## Features
- Joint implementation of **TEAM** (PGA prediction) and **TEAM-LM** (magnitude & location estimation).  
- JSON-based configuration for flexible experiments.  
- Training and evaluation pipelines with TensorBoard logging.  
- Baseline implementations for comparison (magnitude & PGA).  
- Scripts for downloading and processing early warning datasets (Japan, Chile, Italy).  

---

## Installation
We recommend using **conda**:

```bash
conda create -n team python=3.9
conda activate team
pip install -r requirements.txt
```

Notes:
- Requires Python 3.7+.  
- GPU support for TensorFlow **must be installed separately** if needed.  

---

## Usage

### Training
Models are configured via JSON files located in:
- `pga_configs/` → PGA estimation  
- `magloc_configs/` → Magnitude & location estimation  

Example:
```bash
python train.py --config configs/magloc_example.json
```

Options:
- `--test_run` → run with a few data points for quick debugging.  
- Training outputs: model weights and TensorBoard logs (`/logs/scalars`).  

### Evaluation
Evaluate a trained model:
```bash
python evaluate.py --experiment_path [WEIGHTS_PATH]
```

Options:
- `--pga` → evaluate PGA estimation.  
- `--head_times` → evaluate warning times.  
- `--test` → run on test set (default is dev set).  

Evaluation outputs include:
- Performance metrics (R², RMSE, MAE for magnitude/PGA; hypocentral & epicentral errors for location).  
- Plots and prediction files.  

---

## Datasets
Supported datasets:
- **West Sumatra** → requires permission from BMKG.  
- **Italy** → [GFZ Data Service, DOI: 10.5880/GFZ.2.4.2020.004](https://doi.org/10.5880/GFZ.2.4.2020.004)  
- **Chile** → [GFZ Data Service, DOI: 10.5880/GFZ.2.4.2021.002](https://doi.org/10.5880/GFZ.2.4.2021.002)  
- **Japan** → Download via `japan.py` (requires NIED account).  

Example:
```bash
python japan.py --action download_events --catalog resources/kiknet_events --output [OUTPUT FOLDER]
python japan.py --action extract_events --input [DATA FOLDER] --output [HDF5 OUTPUT PATH]
```

Supports **parallel sharding** for large-scale extraction.

---

## Baselines
Reference baseline implementations:
- `mag_baselines.py` → magnitude estimation  
- `pga_baselines.py` → PGA estimation  

Config examples are available in:
- `mag_baseline_configs/`  
- `pga_baseline_configs/`  

---

## Repository Structure
```
├─ configs/              # Example config files
├─ pga_configs/          # Configs for PGA estimation
├─ magloc_configs/       # Configs for magnitude/location estimation
├─ data/                 # Data storage (raw, processed)
├─ scripts/              # Utility scripts
├─ models/               # Model definitions (TEAM, TEAM-LM)
├─ train.py              # Training entry point
├─ evaluate.py           # Evaluation script
├─ requirements.txt      # Python dependencies
└─ README.md
```

---

## Citation
When using **TEAM** or **TEAM-LM**, please cite the software and key publications:

```bibtex
@misc{munchmeyer2021softwareteam,
  doi = {10.5880/GFZ.2.4.2021.003},
  author = {M{"u}nchmeyer, Jannes and Bindi, Dino and Leser, Ulf and Tilmann, Frederik},
  title = {TEAM – The transformer earthquake alerting model},
  publisher = {GFZ Data Services},
  year = {2021},
  note = {v1.0},
  copyright = {GPLv3}
}

@article{munchmeyer2020team,
  title   = {The transformer earthquake alerting model: A new versatile approach to earthquake early warning},
  author  = {M{"u}nchmeyer, Jannes and Bindi, Dino and Leser, Ulf and Tilmann, Frederik},
  journal = {Geophysical Journal International},
  year    = {2020},
  doi     = {10.1093/gji/ggaa609}
}

@article{munchmeyer2021teamlm,
  title   = {Earthquake magnitude and location estimation from real time seismic waveforms with a transformer network},
  author  = {M{"u}nchmeyer, Jannes and Bindi, Dino and Leser, Ulf and Tilmann, Frederik},
  journal = {Geophysical Journal International},
  year    = {2021},
  doi     = {10.1093/gji/ggab139}
}
```

---

## Acknowledgements
- Original TEAM & TEAM-LM implementation: [https://dataservices.gfz-potsdam.de](https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=b028d9d5-832e-11eb-9603-497c92695674).  
- This repository modifies the codebase for **TensorFlow 2.15 compatibility** and additional research experiments.  

---
