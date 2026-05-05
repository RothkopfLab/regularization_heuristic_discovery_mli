# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import ROOT, filter_divergent_chains

sns.set_theme()
sns.set_style("whitegrid")
# %%
n_experiments = 100
num_chains = 10
num_warmup = 4000
num_samples = 20000

prior_conditions = [
    "none",
    "direction",
    "equal_weighing",
    "ranking_sort",
    "single_cue",
]
# %%

idata_dict = {}
env_conditions = ["none", "direction", "ranking"]
reg_conditions = ["unregularized", "regularized"]
#%%
# this cell only works is inference data exists. If not, run the inference script to generate it. This cell loads the inference data and filters out divergent chains.
for env_condition in env_conditions:
    for prior_condition in prior_conditions:
        for reg_condition in reg_conditions:
            (
                idata_dict[env_condition + "-" + prior_condition + "-" + reg_condition],
                del_chains,
            ) = filter_divergent_chains(
                az.from_netcdf(
                    filename=ROOT / f"inference_data/{env_condition}_{reg_condition}_{prior_condition}_{num_chains}chains_{num_samples}samples.nc",
                ),
                0.5,
            )
            print(
                f"Loaded {num_chains-len(del_chains)} chains for prior condition {prior_condition} and env condition {env_condition} {reg_condition}."
            )


# %%
# load results directly
# plot comparison and run if file not found
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
letters = np.array(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]).reshape(2, 3)
for col_idx, env_condition in enumerate(env_conditions):
    for row_idx, regularized in enumerate([False, True]):
        reg = "regularized" if regularized else "unregularized"
        try:
            comparison = pd.read_csv(
                ROOT / f"inference_data/comparison_priors_{env_condition}-{reg}.csv",
                index_col=0,
            )
        except FileNotFoundError:
            keys = [
                key
                for key in idata_dict.keys()
                if (key.split("-")[0] == env_condition and key.split("-")[-1] == reg)
            ]
            comparison = az.compare(
                {key: idata_dict[key] for key in keys}, var_name="answer_probs"
            )
            pd.DataFrame(comparison).to_csv(
                ROOT/f"inference_data/comparison_priors_{env_condition}-{reg}.csv"
            )
        comparison.index = [idx[1] for idx in comparison.index.str.split("-")]
        az.plot_compare(comparison, ax=axs[row_idx, col_idx])
        axs[row_idx, col_idx].set_title(
            f"{letters[row_idx, col_idx]} {'BMI' if regularized else 'MI'}; Cue condition: {env_condition}",
            fontsize=12,
        )
        axs[row_idx, col_idx].set_ylabel("Prior structure")
plt.tight_layout()
plt.savefig(ROOT / "figures"/ "comparison_priors.pdf", bbox_inches="tight")
# %%
