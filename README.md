# PiLA: A PPG-Based Intelligent Framework for Cardiac Disease Diagnosis

This repository provides the official implementation of **PiLA (Physiological-informed Learning Architecture)**, a deep learning framework for PPG-based intelligent cardiac disease diagnosis.

PiLA is an end-to-end framework designed for medical artificial intelligence research. It supports multimodal physiological signal modeling, pretrained weight transfer, and downstream multi-task diagnostic learning.

---

## Framework Overview

PiLA aims to fully exploit physiological information embedded in PPG and its derived signals (e.g., VPG and APG).  
Through a unified deep learning architecture, PiLA enables:

- Representation learning from physiological signals  
- Transfer and freezing of pretrained model weights  
- Downstream multi-task cardiac disease diagnosis (e.g., aortic stenosis and aortic regurgitation)

The framework emphasizes reproducibility and engineering rigor for medical AI research.

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
├── requirements.txt
└── README.md
```

---

## Data Splitting Strategy

For downstream tasks, the dataset is split using stratified sampling to preserve the joint distribution of disease labels across subsets.

The dataset is divided as follows:

- Training set: 64%  
- Validation set: 16%  
- Test set: 20%

Stratification is performed using the joint (AS, AR) label pairs.

---

## Training Workflow

### 1. Pretraining Stage

The pretraining stage is first conducted to learn general physiological representations.

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
├── requirements.txt
└── README.md
```

---

### 2. Downstream Diagnosis Training

After pretraining, the corresponding model weights are loaded and fine-tuned for downstream multi-task cardiac diagnosis.

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
├── requirements.txt
└── README.md
```

During this stage, selected layers of the pretrained network can be frozen to improve training stability.

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
- Due to data privacy constraints, the raw datasets are not publicly available.
- All reported results are obtained using strictly separated training, validation, and independent test sets.

---

## Citation

If you find this work useful, please cite the corresponding paper (to be updated upon public release).
