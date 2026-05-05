# %%
# Fit the models for the priors comparison
import json

import arviz as az
import numpy as np
import numpyro

from src.bayesian_models import ideal_observer_model
from src.utils import data_to_arrays, get_model_predictions, run_mcmc_inference

numpyro.set_platform("cpu")
numpyro.set_host_device_count(16)
from src.utils import ROOT

# %% set params
env_condition = "none"
regularized = True

n_experiments = 100
num_chains = 10
num_warmup = 4000
num_samples = 20000

conditions = [
    "none",
    "direction",
    "equal_weighing",
    "ranking_sort",
    "single_cue",
]


# %% set data and model paths
DATA_PATHS = {
    "direction": ROOT/ "data/ranking_False_direction_True.json",
    "ranking": ROOT/ "data/ranking_True_direction_False.json",
    "none": ROOT/ "data/ranking_False_direction_False.json",
}

if regularized:
    MODEL_PATHS = {
        "direction": ROOT/ "trained_networks/alpha_direction_cross_128_4_0.pth",
        "ranking": ROOT/ "trained_networks/alpha_ranking_cross_128_4_0.pth",
        "none": ROOT/ "trained_networks/alpha_cross_128_4_0.pth",
    }
else:
    MODEL_PATHS = {
        "direction": ROOT/ "trained_networks/pretrained_direction_cross_128_4_0.pth",
        "ranking": ROOT/ "trained_networks/pretrained_ranking_cross_128_4_0.pth",
        "none": ROOT/ "trained_networks/pretrained_cross_128_4_0.pth",
    }

# Load experimental data
data = json.load(open(DATA_PATHS[env_condition]))
model_path = MODEL_PATHS[env_condition]


# %% get model predicitons
inputs = []
targets = []
for i in range(1, n_experiments + 1):
    inp, _, _, tar, _ = data_to_arrays(data, i)
    inputs.append(inp)
    targets.append(tar)
inputs = np.array(inputs)
targets = np.array(targets)
predicted_probabilities = get_model_predictions(model_path, inputs, targets)


# %% method to run model fitting
def run_model_across_conditions(
    inputs,
    targets,
    predicted_probabilities,
    conditions,
    num_warmup,
    num_samples,
    num_chains,
):
    for condition in conditions:
        print(f"Running inference for condition: {condition}")
        result = run_mcmc_inference(
            ideal_observer_model,
            inputs,
            targets,
            predicted_probabilities,
            condition,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        reg = "regularized" if regularized else "unregularized"
        az.to_netcdf(
            result,
            filename=ROOT / f"inference_data/{env_condition}_{reg}_{condition}_{num_chains}chains_{num_samples}samples.nc",
        )


# %% run model fitting
run_model_across_conditions(
    inputs,
    targets,
    predicted_probabilities,
    conditions,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
)
