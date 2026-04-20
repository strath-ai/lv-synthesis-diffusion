# lv-synthesis-diffusion

A Conditional Diffusion Approach for LV Distribution Networks

A PyTorch-based framework for generating realistic low-voltage (LV) electricity load profiles using conditional diffusion models. This tool enables the synthesis of daily active and reactive power demand that preserves temporal patterns and inter-substation coherence, making it suitable for applications such as power system planning, congestion analysis, and scenario generation under increasing low-carbon technology adoption.

## 📚 Citation

If you found this work useful, please cite accordingly:

```bibtex
@article{BRASH2026102264,
title = {Coherent load profile synthesis with conditional diffusion for LV distribution network scenario generation},
journal = {Sustainable Energy, Grids and Networks},
pages = {102264},
year = {2026},
issn = {2352-4677},
doi = {https://doi.org/10.1016/j.segan.2026.102264},
url = {https://www.sciencedirect.com/science/article/pii/S2352467726001463},
author = {Alistair Brash and Junyi Lu and Bruce Stephen and Blair Brown and Robert Atkinson and Craig Michie and Fraser MacIntyre and Christos Tachtatzis},
keywords = {Load modelling, Power systems modelling, Neural network applications, Generative modelling},
abstract = {Limited visibility of distribution network power flows at the low voltage level presents challenges to both distribution network operators from a planning perspective and distribution system operators from a congestion management perspective. More representative loads are required to support meaningful analysis of LV substations; otherwise, such analysis risks misinforming future decisions. Traditional load profiling relies on typical profiles, oversimplifying substation-level complexity. Generative models have attempted to address this through synthesising representative loads from historical exemplars; however, while these approaches can approximate load shapes to a convincing degree of fidelity, analysis of the co-behaviour between substations is limited, which ultimately impacts higher voltage level network operation. This limitation will become even more pronounced with the increasing integration of low-carbon technologies, as estimates of base loads fail to capture load diversity. To address this gap, Conditional Diffusion models for synthesising daily active and reactive power profiles at the low voltage distribution substation level are proposed. The evaluation of fidelity is demonstrated through conventional metrics capturing temporal and statistical realism, as well as power flow modelling. Multiple models are proposed to handle varying levels of data availability, ranging from unconditional synthesis to an informed generation driven by metadata and daily statistics. The results show synthesised load profiles are plausible both independently and as a cohort in a wider power systems context. The Conditional Diffusion model is benchmarked against naive and commonly used generative models to demonstrate its effectiveness in producing realistic scenarios on which to base sub-regional power distribution network planning and operations.}
}
```

## ⚙️ Setup

This project uses a micromamba/conda environment.

### Create environment

```bash
micromamba create -f environment.yml
micromamba activate diffusionwork


## 🔹 Models

### LVGenU — Unconditional
Generates load profiles with no external inputs.  
Learns general patterns directly from historical data.

### LVGenWC — Weather & Calendar Conditioned
Generates load profiles conditioned on:
- Weather data  
- Calendar features  
- Customer information  

### LVGenWCS — Extended Conditioning
Same as LVGenWC, but also includes for both active and reactive power:
- Daily minimum 
- Daily mean  
- Daily maximum  

---

## 🏋️ Training

Each model has its own training script in `Training/`.

### Run Training

    cd Training
    python LVGenU_Train.py

(Replace with `LVGenWC_Train.py` or `LVGenWCS_Train.py` as needed.)

### Output

Checkpoints are saved to:

    Training/results/<MODEL>/T.../XXXX.pkl

Example:

    Training/results/LVGenU/T200_beta00.0001_betaT0.02/10000.pkl

---

## 🔍 Inference (Generation)

Each model has its own inference script in `Inference/`.

### Run Inference

    cd Inference
    python LVGenU_Inference.py --ckpt_iter 10000

Or use the latest checkpoint:

    python LVGenU_Inference.py --ckpt_iter max

### Output

Generated samples are saved to:

    Inference/results/<MODEL>/T.../

Example:

    Inference/results/LVGenU/T200_beta00.0001_betaT0.02/

---
