# CONIA
Classification Optimization Node Injection Attack on Graph Neural Networks


This repository is our Pytorch implementation of our paper:
Classification Optimization Node Injection Attack on Graph Neural Networks


# Requirements
deeprobust==0.2.10
matplotlib==3.7.1
numpy==1.26.4
pandas==1.5.1
scikit_learn==1.1.2
scipy==1.13.0
torch==1.12.1+cu113
torch_geometric==2.5.3
torch_sparse==0.6.14
tqdm==4.64.1

# RUN CODE
## please simple run the following
python attack_main.py --dataset cora --m 5 --r 3 --ceta 1 --injection_ratio 0.03 --epoch_sec 700
