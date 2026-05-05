
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.utils import ROOT

sns.set_style("whitegrid")
# %%
def get_list(group, condition_name, model_name): 
    file_names = []
    save_path = ROOT / f'results/vary_params_group{group}_{model_name}{condition_name}.pth'
    
    if group == 1: #cross
        num_hiddens = ["32", "64", "128"] # TODO Add  
        num_features = ["2", "4", "6"]
        for num_feature in num_features:
            for num_hidden in num_hiddens:
                file_name = ROOT / f"trained_models/{model_name}{condition_name}_cross_{num_hidden}_{num_feature}_0.pth"
                file_names.append(file_name)
    
    return file_names, save_path

# %% Get the variable name and value from the file name
def get_var(file_name):
    parts = file_name.split('_')
    if group == 1:
        num_hidden = int(parts[-3])
        num_feature = int(parts[-2].split('.')[0])
        var = "cross"
        value = f"{num_hidden} hidden, {num_feature} features"
    return var, value




# %% Performance plots
""""
This creates performance plots for all trained models.
"""
groups = [1]
condition_names = ["", "_direction","_ranking",]
model_names = ["pretrained", "alpha"]
group_to_name = {1: "cross", 2: "var", 3: "theta", 4: "eta", 5: "w-std"}
letters = [chr(ord('a') + i) for i in range(6)]

default_values = {1: "64 hidden, 4 Features", 2: "0.01", 3: "1.0", 4: "2.0", 5: "1.0"}
fig, axes = plt.subplots(2, 3, figsize=(5, 3.5), sharey=True, sharex=True)
axes = axes.reshape(2, 3)
subplot_index = 1
for row_index, model_name in enumerate(model_names):
     # Adjusted for two figures
    handles, labels = [], []
    for _, group in enumerate(groups):
        row_handles, row_labels = [], []
        legend_added = False
        
        for col_index, condition_name in enumerate(condition_names):
            ax = axes[row_index, col_index]

            file_names, save_path = get_list(group, condition_name, model_name)
            _, performance_bmi, _ = torch.load(save_path)
            means_bmi = performance_bmi.mean(1).mean(1)
            for i, file_name in enumerate(file_names):
                var, value = get_var(file_name)
                color_idx = 0 if "32" in file_name else 1 if "64" in file_name else 2
                color = sns.color_palette("Set2")[color_idx]
                linestyle = '-' if "_2_" in file_name else '--' if "_4_" in file_name else ':'
                value =value.replace("hidden", "units")
                line, = ax.plot(
                    np.arange(1, 11),
                    means_bmi[i],
                    label=f"{value}",
                    color=color,
                    linestyle=linestyle
                )
            
            ax.set_ylim(0.5, 1)
            ax.set_xlim(1, 10)
            ax.set_xlabel('Trial')
            ax.set_xticks(np.arange(2, 12, 2))
            if col_index == 0:
                ax.set_ylabel('Accuracy')
            condition_str = condition_name.replace("_", "")
            if condition_str == "":
                condition_str = "none"
            model_plot_title = "BMI" if model_name == "alpha" else "MI"
            ax.set_title(f"({letters[subplot_index - 1]}) {model_plot_title};\nCue: {condition_str}", fontsize=11)
            subplot_index += 1
fig.legend(axes[0,0].get_lines(), [line.get_label() for line in axes[0,0].get_lines()], bbox_to_anchor=(0.1, -0.15), loc='center left', title='Model configurations', ncol=2)
plt.tight_layout()
plt.savefig(ROOT / "figures/acc_network_size_features.pdf", bbox_inches='tight')
plt.show()





# %% Plot box plots of gini coefficients

condition_names = ["", "_direction","_ranking"]
model_names = ["pretrained", "alpha"]
letters = ["a", "b", "c", "d", "e", "f"]

fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharey=True, sharex=True)

subplot_index = 1
for j, model_name in enumerate(model_names):
    for i, condition_name in enumerate(condition_names):
        file_names, save_path = get_list(1, condition_name, model_name)

        map_performance, avg_performance, gini_coefficients = torch.load(save_path)

        # Create a DataFrame for box plots
        data_boxplots = []
        for m, file_name in enumerate(file_names):
            parts = file_name.split('_')
            num_features = int(parts[-2].split('.')[0])
            num_hidden = int(parts[-3])
            model_name_plot = f"{num_features} features, {num_hidden} units"
            # Flatten all time steps/trials into one array
            gini_values = gini_coefficients[m].flatten().numpy()
            for gini_val in gini_values:
                data_boxplots.append({
                    "Model": model_name_plot,
                    "Gini Coefficient": gini_val
                })

        df_boxplots = pd.DataFrame(data_boxplots)
        df_boxplots["num_features"] = df_boxplots["Model"].apply(lambda x: x.split(', ')[0])
        df_boxplots["num_hidden"] = df_boxplots["Model"].apply(lambda x: x.split(', ')[1])
        subplot = ax[j, i]

        # Generate boxplots
        box_colors = ['0.8'] * len(df_boxplots['Model'].unique())  # Default color
        middle_index = list(df_boxplots['Model'].unique()).index('4 features, 128 units') #get index of middle boxplot
        if middle_index >=0:
          box_colors[middle_index] = 'olive'
        sns.boxplot(x="num_features", y="Gini Coefficient", hue="num_hidden", data=df_boxplots, palette='Set2', ax=subplot)

        if model_name == "alpha":
            model_plot_title = "BMI"
        else:
            model_plot_title = "MI"

        # Remove the underscore from condition_name
        condition_name_display = condition_name.replace("_", "")
        if condition_name_display == "":
            condition_name_display = "none"
        subplot.set_title(f"({letters[subplot_index - 1]}) {model_plot_title}; Cue condition: {condition_name_display}", fontsize=14)

        subplot.hlines(0.5, 0-0.5, 1-0.5, linestyles='dashed', color='orange', label=f"Max. gini")
        subplot.hlines(0.75, 1-0.5 , 2-0.5, linestyles='dashed', color='orange')
        subplot.hlines(0.833, 2-0.5, 3 -0.5, linestyles='dashed', color='orange')
        subplot.hlines(0, 0-0.5, 3-0.5, color='blue', linestyle='dashed', label=f"Min. gini")
        if subplot_index == 1:
            subplot.legend(loc='upper left', bbox_to_anchor=(3.45, 1))
        else:
            subplot.get_legend().remove()
        subplot.set_xlabel('Number of features')

        subplot_index += 1
        if subplot_index == 7:
            break
    if subplot_index == 7:
        break
plt.tight_layout()
plt.savefig(ROOT / "figures/ginis_network_size_features.pdf", bbox_inches='tight')
plt.show()

# %%
