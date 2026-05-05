# %%
# This script can be used to generate a specific number of experiments (only the
# features, ground-truth weights and targets (without and model responses))
import json

import numpy as np
import torch

from src.environments import PairedComparison


# %%
def generate_trials(
    n_experiments=100,
    n_trials=10,
    n_items=4,
    ranking=False,
    direction=False,
    seed=42,
):
    """
    Generate multiple trials of paired comparison data with specific tensor shapes:
    - inputs, inputs_a, inputs_b: (n_trials, 1, n_items)
    - targets: (n_trials, 1, 1)

    Args:
        n_experiments (int): Number of experiments to generate
        n_trials (int): Length of each experiment
        n_items (int): Number of items to compare
        ranking (bool): Whether to use ranking
        direction (bool): Whether to use direction
        seed (int): Random seed for reproducibility

    Returns:
        dict: Nested dictionary containing all trials and sequences
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_loader = PairedComparison(
        n_items, ranking=ranking, direction=direction, dichotomized=False
    )

    trials_dict = {}

    for exp_id in range(1, n_experiments + 1):
        inputs, targets, inputs_a, inputs_b = data_loader.get_batch(1, n_trials)
        sequence_dict = {}
        for trial_id in range(1, n_trials + 1):
            # Convert tensors to lists while preserving structure
            # Note: seq_id-1 because tensor indexing starts at 0
            sequence_dict[trial_id] = {
                "inputs": inputs[trial_id - 1, 0, :].tolist(),  # (4,)
                "targets": targets[trial_id - 1, 0, 0].item(),  # single value
                "inputs_a": inputs_a[trial_id - 1, 0, :].tolist(),  # (4,)
                "inputs_b": inputs_b[trial_id - 1, 0, :].tolist(),  # (4,)
            }
        trials_dict[exp_id] = sequence_dict
        trials_dict[exp_id]["weights"] = data_loader.weights[0].tolist()
    return trials_dict


def save_trials_to_json(trials_dict, ranking, direction):
    """
    Save trials dictionary to a JSON file.

    Args:
        trials_dict (dict): Dictionary containing trial data
        ranking (bool): Ranking configuration used
        direction (bool): Direction configuration used
    """
    filename = f"ranking_{ranking}_direction_{direction}.json"
    with open(filename, "w") as f:
        json.dump(trials_dict, f, indent=2)


# %%
RANKING = False
DIRECTION = False
N_EXPERIMENTS = 100
N_TRIALS = 10
N_ITEMS = 4
SEED = 42

trials_data = generate_trials(
    n_trials=N_EXPERIMENTS,
    sequence_length=N_TRIALS,
    n_items=N_ITEMS,
    ranking=RANKING,
    direction=DIRECTION,
    seed=SEED,
)
save_trials_to_json(trials_data, RANKING, DIRECTION)
