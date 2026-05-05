from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import torch
import xarray as xr
from jax import random
from numpyro.infer import MCMC, NUTS

from src.model import GRU

ROOT = Path(__file__).parent.parent


def data_to_arrays(data_dict, experiment_id: int):
    experiment_data = data_dict[str(experiment_id)]
    n_trials = len(experiment_data) - 1
    n_items = len(experiment_data["1"]["inputs"])
    inputs = np.zeros((n_trials, n_items))
    inputs_a = np.zeros((n_trials, n_items))
    inputs_b = np.zeros((n_trials, n_items))
    targets = np.zeros(n_trials)
    for seq_id in range(0, n_trials):
        sequence_data = experiment_data[str(seq_id + 1)]
        inputs[seq_id, :] = sequence_data["inputs"]
        targets[seq_id] = sequence_data["targets"]
        inputs_a[seq_id, :] = sequence_data["inputs_a"]
        inputs_b[seq_id, :] = sequence_data["inputs_b"]
    return inputs, inputs_a, inputs_b, targets, experiment_data["weights"]


def get_model_predictions(
    model_path: str, inputs: np.ndarray, targets: np.ndarray, num_hidden=128
) -> np.ndarray:
    """This function can be used to get the prediction of a specific GRU for the
    given inputs and targets.

    Parameters
    ----------
    model_path : str
        Path to the *.pth file
    inputs : np.ndarray
    targets : np.ndarray
    num_hidden : int, optional
        Number of hidden units in the GRU, by default 128

    Returns
    -------
    np.ndarray
        Array of network predictions
    """
    model = GRU(num_inputs=4, num_outputs=1, num_hidden=num_hidden)
    params, _ = torch.load(model_path, map_location="cpu")
    model.load_state_dict(params)
    model.eval()
    if inputs.ndim == 2:
        input_tensor = torch.Tensor(inputs).reshape(
            inputs.shape[0], -1, inputs.shape[1]
        )
        target_tensor = torch.Tensor(targets).reshape(-1, 1, 1)

        predictive_distribution, _, _ = model(input_tensor, target_tensor)
        return predictive_distribution.mean.squeeze().detach().numpy()
    else:
        predictions = []
        for input, target in zip(inputs, targets):
            input_tensor = torch.Tensor(input).reshape(
                input.shape[0], -1, input.shape[1]
            )
            target_tensor = torch.Tensor(target).reshape(-1, 1, 1)

            predictive_distribution, _, _ = model(input_tensor, target_tensor)
            predictions.append(predictive_distribution.mean.squeeze().detach().numpy())
        return predictions


def run_mcmc_inference(model, *args, num_warmup, num_samples, num_chains, seed=10):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    # Generate different seeds for each chain
    rng_keys = random.split(random.PRNGKey(seed), num_chains)
    mcmc.run(rng_keys, *args)

    return az.from_numpyro(mcmc)


def analyze_chain_diagnostics(idata):
    """Analyze MCMC chain diagnostics including divergences and other metrics."""
    n_chains = len(idata.sample_stats.chain)

    divergences = idata.sample_stats["diverging"].values
    divergences_per_chain = divergences.sum(axis=1)

    # Calculate percentage of divergent transitions per chain
    div_percentages = (divergences_per_chain / divergences.shape[1]) * 100
    chain_data = []
    for i in range(n_chains):
        chain_data.append(
            {
                "Chain": i,
                "Divergent_Transitions": int(divergences_per_chain[i]),
                "Divergence_Percentage": float(div_percentages[i]),
            }
        )

    return pd.DataFrame(chain_data)


def filter_divergent_chains(idata, max_divergence_pct=1.0):
    """Filter out all chains with a higher divergence rate than
    max_divergence_pct. max_divergence_pct is in percent. Returns a tuple of the
    filtered idata object and a list of chain ids that were removed from the
    initial idata object."""
    diagnostics = analyze_chain_diagnostics(idata)

    good_chains = diagnostics[
        diagnostics.Divergence_Percentage <= max_divergence_pct
    ].Chain.values

    if len(good_chains) == 0:
        raise ValueError(
            f"All chains exceed maximum divergence percentage of {max_divergence_pct}%!"
            " Consider increasing max_divergence_pct or checking your model"
            " specification."
        )

    # Create a new InferenceData object with only the good chains
    # We need to manually select chains for each group
    filtered_groups = {}

    for group_name in idata._groups_all:
        group = getattr(idata, group_name)
        if isinstance(group, xr.Dataset):
            if "chain" in group.dims:
                # Select only the good chains
                filtered_groups[group_name] = group.isel(chain=good_chains)
            else:
                # If no chain dimension, keep as is
                filtered_groups[group_name] = group

    filtered_idata = az.InferenceData(**filtered_groups)

    removed_chains = diagnostics[
        diagnostics.Divergence_Percentage > max_divergence_pct
    ].Chain.values.tolist()

    return filtered_idata, removed_chains


def print_chain_summary(idata):
    """Print a summary of chain diagnostics."""
    diagnostics = analyze_chain_diagnostics(idata)

    print("Chain Diagnostics Summary:")
    print("-" * 50)
    print("Divergent Transitions per Chain:")
    for _, row in diagnostics.iterrows():
        print(
            f"Chain {int(row['Chain'])}:"
            f" {int(row['Divergent_Transitions'])} divergences"
            f" ({row['Divergence_Percentage']:.2f}%)"
        )

    return diagnostics


def gini(x):
    # https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g
