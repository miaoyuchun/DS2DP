# [TGRS 2022] Hyperspectral Denoising Using Unsupervised Disentangled Spatiospectral Deep Priors (DS2DP)
The official implementation of DS2DP.

## Installation
Clone this repository:
```
git clone git@github.com:miaoyuchun/DS2DP.git
```

The project was developed using Python 3.7.10, and torch 1.12.1.
You can build the environment via pip as follow:

```
pip3 install -r requirements.txt
```

## Running Experiments
We provide code to reproduce the main results on HSI denoising as follows:
```
python DS2DP.py
```

## Citation and Acknowledgement
If you find our work useful in your research, please cite:

```
@article{miao2021hyperspectral,
  title={Hyperspectral Denoising Using Unsupervised Disentangled Spatiospectral Deep Priors},
  author={Miao, Yu-Chun and Zhao, Xi-Le and Fu, Xiao and Wang, Jian-Li and Zheng, Yu-Bang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--16},
  year={2021},
  publisher={IEEE}
}
```
