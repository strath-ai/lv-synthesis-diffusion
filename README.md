# lv-synthesis-diffusion

A Conditional Diffusion Approach for LV Distribution Networks

A PyTorch-based framework for generating realistic low-voltage (LV) electricity load profiles using conditional diffusion models. This tool enables the synthesis of daily active and reactive power demand that preserves temporal patterns and inter-substation coherence, making it suitable for applications such as power system planning, congestion analysis, and scenario generation under increasing low-carbon technology adoption.

## 📚 Citation

If you found this work useful, please cite accordingly:

```bibtex
@Article{brash2025,
  author = {Brash, Alistair and Lu, Junyi and Stephen, Bruce and Brown, Blair and Atkinson, Robert and Michie, Craig and MacIntyre, Fraser and Tachtatzis, Christos},
  title = {Coherent Load Profile Synthesis with Conditional Diffusion for LV Distribution Network Scenario Generation},
  journal = {Sustainable Energy, Grids and Networks},
  year = {2025}
}
```

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
