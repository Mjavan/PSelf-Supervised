# PSelf-Supervised
### Probabilistic Self-Supervised Learning using Cyclical Stochastic Gradient MCMC
This repository provides PyTorch implementation for the paper "A Probabilistic Approach to Self-Supervised Learning using Cyclical Stochastic Gradient MCMC".

![Alt text](Full_image.png)

**Accepted at**:

**Frontiers in Probabilistic Inference: Sampling Meets Learning** Workshop, **ICLR 2025**

**arXiv**: http://arxiv.org/abs/2308.01271

-------------------------------------------

### Installation
 Clone the repo and install using: 
 `pip install .`

### Usage

#### Pretraining
 
To obtain distribution over representations in pretraining phase use `bayesianbyol.py` in `core` module (based on **Bayesian Byol**).

Alternatively, you can take samples from the posterior using **Bayesian SimCLR** `bayesiansimclr.py`.

To trian the model simply run:

`python bayesianbyol.py`

or

`python bayesiansimclr.py`

#### Dataset Splits

For downstream tasks, create different data splits using `split_datasets.py` in `core` module:

`python split_datasets.py`

#### Evaluation
To evaluate the probabilistic representations on an image classification task, use `finetune.py` in the core module.
The function finetunes the pretrained models using samples from the posterior across various data splits. 
Performance is reported on the test set (or validation set, e.g., in ImageNet-10) by marginalizing over the learned representations.
To run:

`python finetune.py`

-----------------------
### Dependencies 
- Pytorch
- Numpy
- Matplotlib
- Scikit-learn
- Seaborn
-----------------------

### Licens
This project is licensed under the MIT License.











