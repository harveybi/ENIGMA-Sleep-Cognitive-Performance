import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_context("paper")  # Adjust global font size
plt.rcParams['font.family'] = 'Arial'

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr

# %%

def replot_plot(y_true, y_pred, save_path=None, fig_title=None, scatter_color=None):
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    corr, corr_p = pearsonr(y_true, y_pred)
    corr_spearman, corr_spearman_p = spearmanr(y_true, y_pred)

    # Create a DataFrame for Seaborn
    data = pd.DataFrame({
        'True Values': y_true,
        'Predicted Values': y_pred
    })

    # Set plot style
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Metrics for legend
    def p_to_superscript(p):
        """Converts p-value to superscript asterisks for legend annotation."""
        if p <= 1e-4:
            asterisks = '****'
        elif p <= 1e-3:
            asterisks = '***'
        elif p <= 1e-2:
            asterisks = '**'
        elif p <= 5e-2:
            asterisks = '*'
        else:
            asterisks = ''
        return asterisks

        # ------------------------- build the label -------------------------
    legend_text = rf"Pearson $r = {corr:.3f}$" + p_to_superscript(corr_p)
    # legend_text = f"Pearson r = {corr:.3f}\np = {corr_p:.4f}"

    # Create lmplot
    if scatter_color is not None:
        lm = sns.lmplot(x='True Values', y='Predicted Values', data=data, ci=None, aspect=1.3, height=7,
                        scatter_kws={'color': scatter_color, 'edgecolor': 'black'},
                        line_kws={'color': scatter_color, 'linewidth': 2})
    else:
        lm = sns.lmplot(x='True Values', y='Predicted Values', data=data, ci=None, aspect=1.3, height=7,
                        scatter_kws={'color': 'blue', 'edgecolor': 'black'}, line_kws={'color': 'blue', 'linewidth': 2})

    lm.ax.text(0.95, 0.95, legend_text, verticalalignment='top', horizontalalignment='right',
               transform=lm.ax.transAxes,
               fontsize=12, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))  # Top-right position

    # Adjustments and display
    lm.set_axis_labels("True Values", "Predicted Values")
    fig_title = fig_title if fig_title else "Actual vs Predicted"
    plt.title(fig_title, fontsize=14, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + fig_title + ".png")
    plt.show()
    plt.close()

    # create jointplot
    if scatter_color is not None:
        join = sns.jointplot(
            x='True Values',
            y='Predicted Values',
            data=data,
            kind='reg',
            height=7,
            scatter_kws={'color': scatter_color, 's': 50, 'alpha': 0.8, 'edgecolor': 'black', 'linewidths': 0.5},
            line_kws={'color': scatter_color, 'linewidth': 4},  # Customize regression line
            marginal_kws={'color': scatter_color, 'bins': 20, 'fill': True}  # Customize marginal distributions
        )
    else:
        join = sns.jointplot(x='True Values', y='Predicted Values', data=data, kind='reg', height=7)

    join.ax_joint.text(0.95, 0.95, legend_text, verticalalignment='top', horizontalalignment='right',
                       transform=join.ax_joint.transAxes, fontsize=14,
                       # bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))  # Top-right position

    # Adjustments and display
    join.set_axis_labels("True Values", "Predicted Values", fontsize=14)
    # set axis tick font
    join.ax_joint.tick_params(axis='both', which='major', labelsize=12)
    # plt.title(fig_title, fontsize=14)  # Add title to the jointplot
    plt.suptitle(fig_title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + fig_title + "_jointplot.png")
        # save as svg
        plt.savefig(save_path + fig_title + "_jointplot.svg", format='svg')
    plt.show()
    plt.close()

# %%
results_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/AutoGluon/EMC/'

target_list = ['Stroop', 'Stroop_rgo_age']
feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
                'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
                'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']

# %%
for feature in feature_list:
    for target in target_list:
        # Check condition: if the target contains '_rgo_age', limit the features
        if '_rgo_age' in target and feature not in ['Sleep_Cov', 'Sleep_Cov_Brain']:
            continue  # Skip this iteration if the feature is not allowed

        # Load results_values.csv
        results_values_path = os.path.join(results_path, target, feature, 'results_values.csv')
        if not os.path.exists(results_values_path):
            print(f"File not found: {results_values_path}")
            continue
        results_values = pd.read_csv(results_values_path)

        EMC_true = results_values['EMC_true'].values
        EMC_pred = results_values['EMC_pred'].values

        save_path = os.path.join(results_path, target, feature)
        if not os.path.exists(save_path):
            print(f"Folder not found: {save_path}")
            continue
        # make plot
        replot_plot(EMC_true, EMC_pred, save_path=save_path + '/', fig_title='Rotterdam',
                    scatter_color='#F3A3A7')
