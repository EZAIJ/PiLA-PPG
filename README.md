# PiLA: A PPG-Based Intelligent Framework for Cardiac Disease Diagnosis

This repository provides the official implementation of **PiLA (Physiological-informed Learning Architecture)**, an end-to-end deep learning framework for PPG-based cardiac disease diagnosis, designed in accordance with reproducibility and transparency standards commonly adopted in medical AI research.

PiLA supports multimodal physiological signal modeling, pretrained representation learning, and downstream multi-task diagnostic training, enabling systematic investigation of physiological-informed learning strategies for cardiovascular disease assessment.

---

## Framework Overview

PiLA aims to fully exploit physiological information embedded in photoplethysmography (PPG) signals and their derived forms, including velocity PPG (VPG) and acceleration PPG (APG).  
Through a unified and modular architecture, PiLA enables:

- Representation learning from multimodal physiological signals  
- Transfer and selective freezing of pretrained model weights  
- Downstream multi-task cardiac disease diagnosis (e.g., aortic stenosis and aortic regurgitation)

The framework emphasizes methodological clarity and engineering rigor to facilitate reproducible medical AI research.

---

## Repository Structure

```
.
├── src/
│   ├── data.py            # Data loading and dataset definitions
│   ├── models.py          # Model architectures (backbone and multi-task models)
│   ├── engine.py          # Training and evaluation pipelines
│   └── utils.py           # Utilities (random seed, checkpointing, etc.)
├── configs/
│   ├── pretrain.yaml      # Configuration for the pretraining stage
│   └── finetune.yaml      # Configuration for downstream training
├── pretrain.py            # Entry point for pretraining
├── finetune.py            # Entry point for downstream diagnosis
├── plot_distribution.py  # Script for label and data distribution visualization
├── requirements.txt
└── README.md
```

---

## Data Splitting Strategy

For downstream diagnostic tasks, the dataset is partitioned using stratified sampling to preserve the joint distribution of disease labels across all subsets.

Specifically, the dataset is split as follows:

- Training set: 64%  
- Validation set: 16%  
- Test set: 20%

Stratification is performed based on the joint label pairs of aortic stenosis (AS) and aortic regurgitation (AR), ensuring consistent class composition across training, validation, and test sets.

---

## Training Workflow

### 1. Pretraining Stage

The pretraining stage is conducted to learn general physiological representations from multimodal signals without task-specific supervision.

```bash
python pretrain.py --config configs/pretrain.yaml
```

### 2. Downstream Diagnosis Training

After pretraining, the learned representations are transferred and fine-tuned for downstream multi-task cardiac disease diagnosis.

```bash
python finetune.py --config configs/finetune.yaml
```

During this stage, selected layers of the pretrained backbone can be frozen to improve training stability and reduce overfitting.

---

## Environment and Dependencies

The main dependencies include:

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- PyYAML

Please refer to `requirements.txt` for the complete dependency list.

---

## Notes

- This codebase is intended for research purposes only.
- Due to data privacy and ethical constraints, the raw datasets are not publicly available.
- All reported results are obtained using strictly separated training, validation, and independent test sets.

---

## Citation

If you find this work useful, please cite the corresponding paper (to be updated upon public release).
