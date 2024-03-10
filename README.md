# PSelf-Supervised
### Probabilistic Self-Supervised Learning using Cyclical Stochastic Gradient MCMC


### Installation
------------------------------------
Install requirements e.g. via 

`pip install -r requirements.txt`

Install `core` module via:

`pip install .`

## Usage
 
To obtain samples from posterior distribution in pretraining phase use `bayesianbyol.py` in `core` module that is based on BayesianByol. Likewise you can take samples from posterior using BayesianSimCLR via `bayesiansimclr`.
To trian the model simply run:

`python bayesianbyol.py`

or

`python bayesiansimclr.py`

For downstream task make different splits of data using `split_datasets.py` in `core` simply via:

`python split_datasets.py`






