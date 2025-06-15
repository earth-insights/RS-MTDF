# RS-MTDF: Multi-Teacher Distillation and Fusion for Remote Sensing Semi-Supervised Semantic Segmentation
This is the official PyTorch implementation of the RS-MTDF method.

## Environment Preparation

First, create a new Conda environment using the `environment.yaml`  provided in the repository:

```bash
conda env create -f environment.yaml
```
Once the environment is created, activate it using the following command:
```bash
conda activate Semi
```
## Data and Pretrained Model Preparation

### Download Datasets

- **LoveDA**: [Train](https://zenodo.org/records/5706578/files/Train.zip?download=1) | [Val](https://zenodo.org/records/5706578/files/Val.zip?download=1)
- **ISPRS Potsdam**: [Data](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
- **DeepGlobe Land Cover**: [Data](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

Expected directory structure:

```bash
./dataset
├── loveda
├── potsdam
├── deepglobe
```
### Data Splits
We follow the experimental setting from [DWL](https://github.com/zhu-xlab/RS-DWL/tree/main/dataloader). After cropping and splitting, the structure should be:
```
├── splits
    ├── ./dataname(eg. loveda)
        ├── 1
            ├── labeled.txt
            ├── unlabeled.txt
        ├── 5
            ├── labeled.txt
            ├── unlabeled.txt
        ├── 10
            ├── labeled.txt
            ├── unlabeled.txt
        ├── test.txt
        ├── val.txt
```
### Pre-trained Encoders
Download the following pretrained models and place them in the ./pretrained directory:

[DINOv2-Small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | [DINOv2-Base](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | [Clip-L](https://huggingface.co/openai/clip-vit-large-patch14)

Expected directory structure:
```
├── ./pretrained
    ├── dinov2_small.pth
    ├── dinov2_base.pth
    ├── ViT-L-14.pt
```
## 🚀 Training
Once the datasets and pretrained models are prepared, start training with:
```bash
sh scripts/train.sh <num_gpu> <port>
```
You can modify `scripts/train.sh` to change the training settings, and adjust the config file for learning rate and other hyperparameters.

## Eval Process
All evaluation scripts are located in the `eval/` directory.

## citation
If you find this project helpful, please consider citing:
``` bibtex
@misc{song2025rsmtdfmultiteacherdistillationfusion,
      title={RS-MTDF: Multi-Teacher Distillation and Fusion for Remote Sensing Semi-Supervised Semantic Segmentation}, 
      author={Jiayi Song and Kaiyu Li and Xiangyong Cao and Deyu Meng},
      year={2025},
      eprint={2506.08772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.08772}, 
}
```
