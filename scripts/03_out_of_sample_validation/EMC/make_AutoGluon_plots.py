import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/lib/')  # change to the path where the lib folder is located

import argparse
import pandas as pd

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='AutoGluon, out-of-sample validation')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
    - Brain: Uses brain features only.
    - Sleep_Cov_Brain: Uses sleep features, covariates, and brain features.
    - Sleep_Cov_Brain_Shuffle: Uses sleep features, covariates, and brain features (shuffled).
    - Sleep_APOE: Uses sleep features and APOE4.
    - Sleep_APOE_Shuffle: Uses sleep features (shuffled) and APOE4.
    - Sleep_Cov_APOE: Uses sleep features, covariates, and APOE4.
    - Sleep_Cov_APOE_Shuffle: Uses sleep features, covariates (shuffled), and APOE4.
    - Sleep_Cov_Brain_APOE: Uses sleep features, covariates, brain features, and APOE4.
    - Sleep_Cov_Brain_APOE_Shuffle: Uses sleep features, covariates, brain features (shuffled), and APOE4.
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
    - Stroop_rgo_age: Stroop_rgo_age
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting AutoGluon out-of-sample validation pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
results_save_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/AutoGluon/EMC/{target}/{feature_comb}/'

df_results_values = pd.read_csv(results_save_path + 'results_values.csv')

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def test_perform_plot(results_values, fig_title=None, scatter_color=None, save_path=None):

    y_true = results_values['EMC_true']
    y_pred = results_values['EMC_pred']

    # print range of y_true and y_pred
    print(f"\ny_true range: {y_true.min()} - {y_true.max()}\n")
    print(f"y_pred range: {y_pred.min()} - {y_pred.max()}\n")

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
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
    sns.set_context("paper")  # Adjust global font size

    # Metrics for legend
    legend_text = f"Pearson r = {corr:.3f}\np = {corr_p:.4f}"

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
    fig_title = fig_title + " - Actual vs Predicted" if fig_title else "Actual vs Predicted"
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
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))

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

    # Print metrics
    print(f"R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, Pearson r: {corr:.3f}, Spearman r: {corr_spearman:.3f}")

# %%
test_perform_plot(df_results_values, fig_title=f'Rotterdam', scatter_color='#F3A3A7', save_path=results_save_path)
