# Transformer Models for Seismic Facies Segmentation

This repository contains the configuration files used in the experiments conducted for the Master's thesis _"An Evaluation of Transformer Models for Seismic Facies Segmentation"_ by Gabriel Borges Gutierrez, defended at the Institute of Computing, University of Campinas (UNICAMP), in 2025.

## 📄 Thesis Summary

The thesis investigates the use of Transformer-based architectures for seismic facies segmentation, comparing them against strong CNN-based baselines under identical training conditions. Five models were evaluated:

- CNN-based:
  - DeepLab V3
  - DeepLab V3+

- Transformer-based:
  - SegFormer
  - Segmenter
  - SETR-PUP

All models were trained on the F3 and Seam AI datasets using the [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) framework. Additionally, a hyperparameter optimization pipeline was applied to the best-performing model (SETR-PUP), using the author's in-house framework.


## 🛠️ Requirements

- Python 3.8+
- PyTorch
- MMCV
- MMsegmentation
- CUDA 11.3+

Install MMsegmentation following the official instructions:  
👉 https://mmsegmentation.readthedocs.io/en/latest/get_started.html

## 📊 Datasets

### F3 Dataset
Seismic volume from the Netherlands F3 block. Class merging and data splits follow **Alaudah et al. (2019)**.

### Seam AI Dataset
Public dataset from the SEAM AI Challenge (NZPM). Data split follows **Tolstaya and Egorov (2022)**.

## ⚙️ Training Details

- Input size: 512 × 512
- Epochs: 1000
- Batch size: 2
- Optimizer: SGD
- Learning rate: 0.001
- Loss: Weighted Cross-Entropy
- Pretrained weights: ImageNet-21k

## 📈 Results

Quantitative and qualitative results are reported in the thesis, including model comparisons on mIoU, pixel accuracy, and class accuracy. Visual results and ablation studies are also included.

### 📊 F3 Dataset Results (1000 Epochs)

| **Model**       | **mIoU** | **Pixel Accuracy (PA)** | **Mean Class Accuracy (MCA)** |
|------------------|--------:|-------------------------:|-------------------------------:|
| DeepLab V3       | 0.7781  | 0.9392                   | 0.8742                         |
| DeepLab V3+      | 0.7748  | 0.9365                   | 0.8649                         |
| SegFormer        | 0.7268  | 0.9182                   | 0.8198                         |
| Segmenter        | 0.7800  | 0.9353                   | 0.8815                         |
| **SETR (PUP)**   | 0.7849  | 0.9338                   | 0.8828                         |

### 📊 Seam AI Dataset Results (1000 Epochs)

| **Model**       | **mIoU** | **Pixel Accuracy (PA)** | **Mean Class Accuracy (MCA)** |
|------------------|--------:|-------------------------:|-------------------------------:|
| DeepLab V3       | 0.7128  | 0.9236                   | 0.833                         |
| DeepLab V3+      | 0.7095  | 0.9211                   | 0.8244                        |
| SegFormer        | 0.6406  | 0.8874                   | 0.712                         |
| Segmenter        |0.7012   | 0.9167                   | 0.8102                        |
| **SETR (PUP)**   | 0.7605  | 0.9377                   | 0.8407                        |


You can read the full document here:  
👉 [Thesis PDF](https://github.com/GabrielBG0/Evaluation-of-Transformers-Seismic-Facies-Segmentation/blob/main/Transformer%20Models%20for%20Seismic%20Facies%20Segmentation.pdf)

## 📜 Citation

If you use this repository or reproduce part of the thesis, please cite:

```bibtex
@mastersthesis{gutierrez2025transformers,
  title     = {An Evaluation of Transformer Models for Seismic Facies Segmentation},
  author    = {Gabriel Borges Gutierrez},
  school    = {University of Campinas (UNICAMP)},
  year      = {2025}
}
```
## 📫 Contact

ORCID: [0009-0008-4200-8963](https://orcid.org/0009-0008-4200-8963)

Lattes: [https://lattes.cnpq.br/6015137863078458](https://lattes.cnpq.br/6015137863078458)

Email: [Gabriel.bgs00@gmail.com](mailto:gabriel.bgs00@gmial.com)
