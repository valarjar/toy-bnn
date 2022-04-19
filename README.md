# Hands-on Bayesian Neural Networks

Three toy examples for the tutorial "Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users". 

## Dependancies

The code depends on: 

- numpy (tested with version 1.19.2), 
- pytorch (tested with version 1.8.1),
- torchvision (tested with version 0.9.1),
- matplotlib (tested with version 3.1.1),
- scipy (tested with version 1.3.1), 
- pandas (tested with version 1.0.2),
- pyro (tested with version 1.6.0),
- seaborn (tested with version 0.10.0),

and libraries from the base python distribution: argparse, os and time.

It has been tested with python 3.6.9.


## Example 1: Bayesian MNIST

Bayesian MNIST is just a hello world project showing how a BNN can be implemented to perform classification on MNIST.

## Usage

The project is split into multiple files:

- dataset.py implement a few routines to filter out the mnist dataset, allowing us to train the model without one digit, as it will be presented later to the model to see how it reacts.
- viModel.py implement the variational inference layers and model we are using.
- viExperiment.py is the script running the actual experiment. It can be called with the -h option to get a contextual help message:

	python viExperiment.py -h


## Example 2: Sparse Measure

Sparse Measure aims at showing how different learning strategies can be implemented for a BNN.

## Usage

The project is split into multiple files:

- dataset.py implement the primitives to generate the dataset.
- experiment.py contain a small script to run the experiment which generated figure 16 in the paper
- viModel.py implement the variational inference layers and model we are using.
- viExperiment.py is the script running the actual experiment. It can be called with the -h option to get a contextual help message:

	python viExperiment.py -h


## Example 3: Paperfold

Paperfold illustrates different inference methods for BNm, as well as some of their benefits and limitations, on a small model with 8 parameters (To make it easier to plot the actual samples from the posterior).

## Usage

The project is split into multiple files. A first series of modules define the models:

- numpyModel contain the model implemented using numpy primitives. This allows to use the samples generated by the different models.
- pyroModel contain the model implemented using pyro primitives. This is used mainly for mcmc based inference.
- torchModel contain a point estimate version of the model (based on maximum likelyhood), it was not used in the final experiment.
- viModel contain the MAP point estimate version of the model and a mean field gaussian based version (for variational inference).

Then, a series of experiment scripts use an inference method to get the posterior. To provided a uniform interface for the next module in the pipeline, each of those scripts generate a pickle file containing samples from the posterior:

- mcmc_experiment generate those samples using a state of the art MCMC sampler from pyro.
- vi_experiment generate those samples using either the MAP point estimate model or the mean field gaussian model. Ensembling can be enable using a command line switch.

Finally, the results can be analysed by using the plots script.

The experiment and plotting scripts provide contextual help when called with the -h option:

	python module_name.py -h


## Citation

@article{DBLP:journals/corr/abs-2007-06823,
author    = {Laurent Valentin Jospin and
			Wray L. Buntine and
			Farid Boussa{\"{\i}}d and
			Hamid Laga and
			Mohammed Bennamoun},
title     = {Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users},
journal   = {CoRR},
volume    = {abs/2007.06823},
year      = {2020},
url       = {https://arxiv.org/abs/2007.06823},
archivePrefix = {arXiv},
eprint    = {2007.06823},
timestamp = {Tue, 21 Jul 2020 12:53:33 +0200},
biburl    = {https://dblp.org/rec/journals/corr/abs-2007-06823.bib},
bibsource = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-2007-06823,
author    = {Laurent Valentin Jospin and
			Wray L. Buntine and
			Farid Boussa{\"{\i}}d and
			Hamid Laga and
			Mohammed Bennamoun},
title     = {Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users},
journal   = {CoRR},
volume    = {abs/2007.06823},
year      = {2020},
url       = {https://arxiv.org/abs/2007.06823},
archivePrefix = {arXiv},
eprint    = {2007.06823},
timestamp = {Tue, 21 Jul 2020 12:53:33 +0200},
biburl    = {https://dblp.org/rec/journals/corr/abs-2007-06823.bib},
bibsource = {dblp computer science bibliography, https://dblp.org}
}


@article{DBLP:journals/corr/abs-2007-06823,
author    = {Laurent Valentin Jospin and
			Wray L. Buntine and
			Farid Boussa{\"{\i}}d and
			Hamid Laga and
			Mohammed Bennamoun},
title     = {Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users},
journal   = {CoRR},
volume    = {abs/2007.06823},
year      = {2020},
url       = {https://arxiv.org/abs/2007.06823},
archivePrefix = {arXiv},
eprint    = {2007.06823},
timestamp = {Tue, 21 Jul 2020 12:53:33 +0200},
biburl    = {https://dblp.org/rec/journals/corr/abs-2007-06823.bib},
bibsource = {dblp computer science bibliography, https://dblp.org}
}