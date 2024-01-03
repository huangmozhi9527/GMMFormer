# GMMFormer: Gaussian-Mixture-Model Based Transformer for Efficient Partially Relevant Video Retrieval

This repository is the official PyTorch implementation of our AAAI 2024 paper [GMMFormer: Gaussian-Mixture-Model Based Transformer for Efficient Partially Relevant Video Retrieval](https://arxiv.org/pdf/2310.05195.pdf).


## Catalogue <br> 
* [1. Getting Started](#getting-started)
* [2. Run](#run)
* [3. Trained Models](#trained-models)
* [4. Results](#results)
* [5. Citation](#citation)



## Getting Started

1\. Clone this repository:
```
git clone https://github.com/haungmozhi9527/GMMFormer.git
cd GMMFormer
```

2\. Create a conda environment and install the dependencies:
```
conda create -n prvr python=3.9
conda activate prvr
conda install pytorch==1.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

3\. Download Datasets: All features of TVR, ActivityNet Captions and Charades-STA are kindly provided by the authors of [MS-SL].


4\. Set root and data_root in config files (*e.g.*, ./Configs/tvr.py).

## Run

To train GMMFormer on TVR:
```
cd src
python main.py -d tvr --gpu 0
```

To train GMMFormer on ActivityNet Captions:
```
cd src
python main.py -d act --gpu 0
```

To train GMMFormer on Charades-STA:
```
cd src
python main.py -d cha --gpu 0
```



## Trained Models

We provide trained GMMFormer checkpoints. You can download them from Baiduyun disk.

| *Dataset* | *ckpt* |
| ---- | ---- |
| TVR | [Baidu disk](https://pan.baidu.com/s/1UK8Lzyc9msmzveIViLSIvg?pwd=qy7s) |
| ActivityNet Captions | [Baidu disk](https://pan.baidu.com/s/1GkV8jTF1SQylJnNu-ba_Ow?pwd=mfkk) |
| Charades-STA | [Baidu disk](https://pan.baidu.com/s/1DaopvMUEcEueSH3Xf6VDbg?pwd=1wu6) |

## Results

### Quantitative Results

For this repository, the expected performance is:

| *Dataset* | *R@1* | *R@5* | *R@10* | *R@100* | *SumR* |
| ---- | ---- | ---- | ---- | ---- | ---- |
| TVR | 13.9 | 33.3 | 44.5 | 84.9 | 176.6 |
| ActivityNet Captions | 8.3 | 24.9 | 36.7 | 76.1 | 146.0 |
| Charades-STA | 2.1 | 7.8 | 12.5 | 50.6 | 72.9 |

## Citation

If you find this repository useful, please consider citing our work:

```
@inproceedings{wang2023gmmformer,
  title={GMMFormer: Gaussian-Mixture-Model Based Transformer for Efficient Partially Relevant Video Retrieval},
  author={Wang, Yuting and Wang, Jinpeng and Chen, Bin and Zeng, Ziyun and Xia, Shu-Tao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

[MS-SL]:https://github.com/HuiGuanLab/ms-sl



