<!--# FedGAT
Official implementation of FedGAT: Generative Autoregressive Transformers Model-Agnostic Federated MRI Reconstruction -->
<hr>
<h1 align="center">
  FedGAT <br>
  <sub>Generative Autoregressive Transformers for Model-Agnostic Federated MRI Reconstruction</sub>
</h1>

<div align="center">
  <a href="https://github.com/Valiyeh" target="_blank">Valiyeh&nbsp;A. Nezhad</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/gelmas07" target="_blank">Gokberk&nbsp;Elmas</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://bilalkabas.github.io/" target="_blank">Bilal&nbsp;Kabas</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/fuat-arslan" target="_blank">Fuat&nbsp;Arslan</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://kilyos.ee.bilkent.edu.tr/~cukur/" target="_blank">Tolga&nbsp;Çukur</a><sup>1,2</sup>

  <br><br>
  <sup>1</sup>UMRAM, <sup>2</sup>Bilkent University
</div>
<hr>

<h3 align="center">[<a href="https://arxiv.org/abs/2502.04521">arXiv:2502.04521</a>]</h3>

Official PyTorch implementation of **FedGAT**, a novel model-agnostic federated learning technique based on generative autoregressive transformers for MRI reconstruction. Unlike conventional federated learning that requires homogeneous model architectures across sites, FedGAT enables flexible collaborations among sites with distinct reconstruction models by decentralizing the training of a global generative prior. This prior captures the distribution of multi-site MRI data via autoregressive prediction across spatial scales, guided by a site-specific prompt. Site-specific reconstruction models are trained using hybrid datasets combining local and synthetic samples. Comprehensive experiments demonstrate that FedGAT achieves superior within-site and across-site reconstruction performance compared to state-of-the-art FL baselines while preserving privacy.

<div align="center">
  <img src="figures/fedgat.png" width="700"/>
</div>

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/icon-lab/FedGAT.git
cd FedGAT

# Create and activate conda environment
conda env create -f environment.yml
conda activate fedgat
```

---

## 📚 Data Preparation

Expected dataset structure:

```
data/
├── Site_0/
│   ├── train/
│   │   └── data/
│   └── val/
│       └── data/
├── Site_1/
│   ├── train/
│   │   └── data/
│   └── val/
│       └── data/
├── Site_2/
│   ├── train/
│   │   └── data/
│   └── val/
│       └── data/
```

Each `train/` and `val/` folder contains MRI images (e.g., `.png` files) for each site.

---

## 🏋️ Training

### Basic Training Command

```bash
torchrun --nproc_per_node=1 train.py \
    --case='multicoil' \
    --depth=16 \
    --bs=16 \
    --ep=500 \
    --fp16=1 \
    --alng=1e-3 \
    --wpe=0.1 \
    --client_num=3 \
    --comm_round=1

```


### Training Parameters

| Parameter        | Description                                       | Default |
|------------------|---------------------------------------------------|---------|
| `--case`         | Dataset type (`'singlecoil'` or `'multicoil'`)      | -       |
| `--client_num`   | Number of federated Sites                       | 3       |
| `--comm_round`   | Number of communication rounds                   | 1       |
| `--depth`        | Model depth                                       | 16      |
| `--bs`           | Batch size                                        | 16      |
| `--ep`           | Number of epochs                                  | 200     |
| `--fp16`         | Mixed precision training                          | 1       |
| `--alng`         | AdaLN gamma                                        | 1e-3    |
| `--wpe`          | Final learning rate ratio at the end of training   | 0.1     |


---

FedGAT will create a `fedGAT_output/` directory to store all checkpoints and logs. You can monitor training by:

- Inspecting `fedGAT_output/log.txt` and `fedGAT_output/stdout.txt`  

If your run is interrupted, simply re-execute the same training command—FedGAT will automatically pick up from the latest `fedGAT_output/ckpt*.pth` checkpoint (see `utils/misc.py`, lines 344–357).


## 📖 Citation

You are welcome to use, modify, and distribute this code. We kindly request that you acknowledge this repository and cite our paper appropriately.

```bibtex
@article{nezhad2025generative,
  title={Generative Autoregressive Transformers for Model-Agnostic Federated MRI Reconstruction},
  author={Nezhad, Valiyeh A and Elmas, Gokberk and Kabas, Bilal and Arslan, Fuat and {\c{C}}ukur, Tolga},
  journal={arXiv preprint arXiv:2502.04521},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This repository uses code from the following projects:
- [VAR (Visual AutoRegressive modeling)](https://github.com/FoundationVision/VAR)
- [VQGAN (Taming Transformers)](https://github.com/CompVis/taming-transformers)


---

Copyright © 2025, ICON Lab.
