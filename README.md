# Regularization more than representational capacity drives heuristic discovery in Bounded Meta-Learned Inference

This repository contains the official implementation of the paper "Regularization more than representational capacity drives heuristic discovery in Bounded Meta-Learned Inference" by Lars C. Reining, Finn Radatz, Nora Cremille and Constantin A. Rothkopf. The paper was accepted for publication at the Annual Meeting of the Cognitive Science Society (CogSci 2026).

## Usage

To run the code, clone the repository and install the required packages using uv:

```bash
git clone https://github.com/RothkopfLab/regularization_heuristic_discovery
cd regularization_heuristic_discovery
uv sync
```

## Repository structure

- `compare_network_sizes_plots.py`: Reproduces Figure 2 and 3 of the paper, comparing the performance of different network sizes and number of features.
- `compare_priors_plots.py`: Reproduces Figure 4 of the paper, comparing the influence of different priors on model fit.
- `fit_bayesian_models.py`: Contains the code for fitting the Bayesian model to the data.
- `generate_experiments.py`: Script to sample paired comparison tasks. The tasks used in the experiment can be found in the `data` folder.
- `data/`: Contains the paired comparison tasks used to train and test the models.
- `figures/`: Contains the generated figures from the paper.
- `inference_data/`: Contains summary statistics of the inference data and can also contain (though not included in the repository) the raw inference data (chains).
- `results/`: cached performances and predictions of the neural network models.
- `src/`:
  - `bayesian_models.py`: Contains the implementation of the Bayesian model and the priors.
  - `environments.py`: Contains the implementation of the paired comparison tasks. Based on the code by Marcel Binz (https://github.com/marcelbinz/HeuristicsFromBMLI/).
  - `model.py`: Contains the implementation of the GRU. Also based on the code by Marcel Binz.
  - `utils.py`: Contains utility functions.
- `trained_models/`: Contains the checkpoints of the trained neural network models. Models prefixed with `alpha` correspond to the models trained with the regularization term, while models prefixed with `pretrained` correspond to the models trained without the regularization term. The first number corresponds to the number of hidden units, while the second number corresponds to the number of features.



