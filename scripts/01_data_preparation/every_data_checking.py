import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

from ridgeplot import ridgeplot
import kaleido
kaleido.get_chrome_sync()
from itertools import combinations
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# %%
# reload utils
import importlib
importlib.reload(utils)

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
result_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'

def describe_and_count(df):
    """
    Describe key numeric columns and count SEX/APOE4 categories.
    Safely skips any columns that are missing.
    """
    df = df.copy()

    # --- numeric summaries ---
    numeric_cols = ['Age_at_Scan', 'Depression_score', 'BMI', 'PSG_Sleep_Dur', 'Self_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Eff']
    existing_numeric = [c for c in numeric_cols if c in df.columns]
    missing_numeric = [c for c in numeric_cols if c not in df.columns]

    if existing_numeric:
        print("Numeric variable summary:")
        print(df[existing_numeric].describe())
    else:
        print(f"No numeric columns present among: {numeric_cols}")

    if missing_numeric:
        print("Missing numeric columns:", ", ".join(missing_numeric))

    # --- SEX counts ---
    if 'SEX' in df.columns:
        df.loc[:, 'SEX'] = df['SEX'].astype('category')
        sex_count = df['SEX'].value_counts().get(2, 0)
        print(f"Number of subjects with SEX value 2: {sex_count}")
    else:
        print("Column 'SEX' not found; skipping SEX summary.")

    # --- APOE4 counts ---
    if 'APOE4' in df.columns:
        apoe_counts = df['APOE4'].value_counts()
        apoe4_count_1 = apoe_counts.get(1, 0)
        apoe4_count_2 = apoe_counts.get(2, 0)
        apoe4_missing_count = df['APOE4'].isna().sum()

        print(f"Number of subjects with APOE4 value 1: {apoe4_count_1}")
        print(f"Number of subjects with APOE4 value 2: {apoe4_count_2}")
        print(f"Number of missing values in APOE4 column: {apoe4_missing_count}")
    else:
        print("Column 'APOE4' not found; skipping APOE4 counts.")


# %%
"""
Ridge plot of 'Age_at_Scan', Stroop interference score, and memory test score
Color maps of all sites
SHIP-Trend: #18476A
EMC: #F3A3A7
VETSA: #E4BDE5
Liege: #56BBB8
KI: PASTEL GREEN #7EA8D3
Pitts: STONE #96D3A1
Juelich: #90A2A5
"""
def ridge_plot(save_path, df, x_variable, y_variable, x_label=None, color_palette=None):
    """
    Make ridge plot of a feature of a dataset with a customizable color palette.

    Parameters:
    - df: DataFrame containing the data.
    - x_variable: The variable to plot on the x-axis.
    - y_variable: The categorical variable for splitting the ridges.
    - x_label: The label for the x-axis.
    - color_palette: Custom color palette for the plot. Can be a Seaborn color palette or a list of colors.
    """
    # Set the seaborn theme for better visualization
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # figsize
    plt.figure(figsize=(6, 8))

    # Set a custom color palette if provided
    if color_palette is None:
        # Default to cubehelix palette if no palette is passed
        color_palette = sns.cubehelix_palette(len(df[y_variable].unique()), rot=-.25, light=.7)

    # Initialize the FacetGrid object with Dataset as the row variable
    g = sns.FacetGrid(df, row=y_variable, hue=y_variable, aspect=6, height=2, palette=color_palette)

    # Plot the KDEs (Kernel Density Estimates)
    g.map(sns.kdeplot, x_variable,
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)

    # Overlay white lines for contrast
    g.map(sns.kdeplot, x_variable, clip_on=False, color="w", lw=2, bw_adjust=.5)

    # Add a reference line at y=0
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define a function to add labels inside each facet
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.1, .2, label, fontweight="bold", color=color, fontsize=18,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, x_variable)

    # Adjust the spacing between plots
    g.figure.subplots_adjust(hspace=-.3)

    # Remove titles and axis labels that may overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")

    if x_label:
        # Change x-axis label size and style
        g.set(xlabel=x_label)

        # Adjust tick and label size using Matplotlib
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=18)  # Change x-axis tick label size
            ax.set_xlabel(x_label, fontsize=20)  # Set x-axis label size and style

    g.despine(bottom=True, left=True)

    # plt.tight_layout()
    # save figure as svg
    plt.savefig(save_path + f'{x_variable}_ridge_plot.svg', format='svg')
    # Show the final ridge plot
    plt.show()
    plt.close()


# def ridge_plot_new(save_path, df, x_variable, y_variable, x_label=None, color_palette="viridis"):
#     """
#     Create a ridge plot using the ridgeplot library with x-axis ticks ranging from 0 to 100.
#
#     Parameters:
#     - save_path: Path to save the plot as an SVG file.
#     - df: DataFrame containing the data.
#     - x_variable: The numerical variable to plot (e.g., "Age_at_Scan").
#     - y_variable: The categorical variable for splitting the ridges (e.g., "Dataset").
#     - x_label: The label for the x-axis (optional).
#     - color_palette: Color scale for the ridge plot (default is "viridis").
#     """
#     # Prepare the data for ridgeplot
#     unique_categories = df[y_variable].unique()
#     category_order = unique_categories
#     data_samples = [df[df[y_variable] == category][x_variable].dropna().values for category in category_order]
#
#     # Define KDE points for consistent x-axis across ridges
#     kde_points = np.linspace(0, 100, 500)  # Set KDE points from 0 to 100
#
#     # Generate the ridge plot
#     fig = ridgeplot(
#         samples=data_samples,
#         bandwidth=4,  # Adjust KDE bandwidth as needed
#         kde_points=kde_points,
#         colorscale=color_palette,
#         colormode="row-index",
#         opacity=0.6,
#         labels=category_order,
#         spacing=5 / 9,  # Adjust ridge spacing
#     )
#
#     # Customize the plot
#     fig.update_layout(
#         height=620,  # 560
#         width=680,  # 800
#         font=dict(
#             family="Arial",  # Set font family to Arial
#             size=18,  # Set font size for all text
#             color="black"  # Set font color to black
#         ),
#         plot_bgcolor="white",
#         xaxis_title=x_label if x_label else x_variable,
#         xaxis_tickvals=np.arange(0, 101, 10),  # Set x-axis ticks from 0 to 100 with step 10
#         xaxis_range=[0, 100],  # Restrict x-axis range to 0-100
#         xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
#         yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
#         showlegend=False,
#     )
#
#     # Save the plot
#     fig.write_image(save_path + f"{x_variable}_ridge_plot.svg")
#     # also save a png version
#     fig.write_image(save_path + f"{x_variable}_ridge_plot.png")
#
#     # Show the plot (optional)
#     fig.show()
def ridge_plot_new(save_path, df, x_variable, y_variable, x_label=None, color_palette="viridis"):
    """
    Create a ridge plot using the ridgeplot library with x-axis ticks ranging from 0 to 100.

    Parameters:
    - save_path: Path to save the plot as an SVG file.
    - df: DataFrame containing the data.
    - x_variable: The numerical variable to plot (e.g., "Age_at_Scan").
    - y_variable: The categorical variable for splitting the ridges (e.g., "Dataset").
    - x_label: The label for the x-axis (optional).
    - color_palette: Color scale for the ridge plot (default is "viridis").
    """
    # -----------------------------
    # Local font-size constants
    # -----------------------------
    FS_TICK = 10
    FS_LABEL = 12
    FS_TITLE = 14  # (kept for consistency; not used unless you add a title)

    # Prepare the data for ridgeplot
    unique_categories = df[y_variable].unique()
    category_order = unique_categories
    data_samples = [df[df[y_variable] == category][x_variable].dropna().values for category in category_order]

    # Define KDE points for consistent x-axis across ridges
    kde_points = np.linspace(0, 100, 500)

    # Generate the ridge plot
    fig = ridgeplot(
        samples=data_samples,
        bandwidth=4,
        kde_points=kde_points,
        colorscale=color_palette,
        colormode="row-index",
        opacity=0.6,
        labels=category_order,
        spacing=5 / 9,
    )

    # Figure size scaled to number of ridges (smaller fonts => smaller base)
    n_cat = len(category_order)
    fig_h = 350  # px, max(360, 120 + 35 * n_cat); 620
    fig_w = 350  # px, 560; 680

    # Customize
    fig.update_layout(
        height=fig_h,
        width=fig_w,
        font=dict(
            family="Arial",
            size=FS_TICK,
            color="black",
        ),
        plot_bgcolor="white",
        xaxis_title=x_label if x_label else x_variable,
        xaxis_title_font=dict(size=FS_LABEL),
        yaxis_title_font=dict(size=FS_LABEL),
        xaxis_tickvals=np.arange(0, 101, 10),
        xaxis_range=[0, 100],
        xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
        showlegend=False,
    )

    fig.write_image(save_path + f"{x_variable}_ridge_plot_new.svg")
    fig.write_image(save_path + f"{x_variable}_ridge_plot_new.png")


def stacked_density_plot(df, x_variable, y_variable, x_label=None, color_palette=None):
    """
    Create a stacked density plot.
    """
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Define color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Plot the KDEs
    plt.figure(figsize=(10, 6))
    for category, color in zip(df[y_variable].unique(), color_palette):
        subset = df[df[y_variable] == category]
        sns.kdeplot(subset[x_variable], fill=True, alpha=0.6, linewidth=2, color=color, label=category)

    # Labeling
    plt.xlabel(x_label if x_label else x_variable, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Stacked Density Plot", fontsize=16)
    plt.legend(title=y_variable, fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close()


def violin_plot(df, x_variable, y_variable, x_label=None, y_label=None, color_palette=None):
    """
    Create a violin plot.
    """
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Define color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=y_variable, y=x_variable, palette=color_palette, inner="box")

    # Labeling
    plt.xlabel(y_variable if y_label is None else y_label, fontsize=14)
    plt.ylabel(x_variable if x_label is None else x_label, fontsize=14)
    plt.title("Violin Plot", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()


def faceted_histograms(df, x_variable, y_variable, x_label=None, color_palette=None):
    """
    Create faceted histograms for the dataset.
    """
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Define color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Create FacetGrid
    g = sns.FacetGrid(df, col=y_variable, col_wrap=3, sharey=False, palette=color_palette, height=4)
    g.map(plt.hist, x_variable, bins=15, color="gray", edgecolor="black", alpha=0.7)

    # Adjust titles and labels
    g.set_titles("{col_name}")
    g.set_axis_labels(x_label if x_label else x_variable, "Count")
    g.set_xticklabels(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()


def boxplot_swarmplot_combo(df, x_variable, y_variable, x_label=None, y_label=None, color_palette=None):
    """
    Create a combination of a boxplot and a swarmplot.

    Parameters:
    - df: DataFrame containing the data.
    - x_variable: The numerical variable for the y-axis.
    - y_variable: The categorical variable for splitting the x-axis.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - color_palette: Custom color palette.
    """
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Define a color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the boxplot
    sns.boxplot(data=df, x=y_variable, y=x_variable, palette=color_palette, width=0.6, showcaps=True,
                boxprops={'zorder': 1})

    # Overlay the swarmplot
    sns.swarmplot(data=df, x=y_variable, y=x_variable, palette=color_palette, size=5, alpha=0.8, zorder=2)

    # Labeling
    plt.xlabel(y_variable if x_label is None else x_label, fontsize=14)
    plt.ylabel(x_variable if y_label is None else y_label, fontsize=14)
    plt.title("Boxplot + Swarmplot Combination", fontsize=16)

    plt.show()


def violin_swarm_plot(df, x_variable, y_variable, x_label=None, y_label=None, color_palette=None):
    """
    Create a combined violin and swarm plot.

    Parameters:
    - df: DataFrame containing the data.
    - x_variable: The variable to plot on the y-axis.
    - y_variable: The categorical variable to separate distributions.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - color_palette: List of colors or Seaborn color palette for the categories.
    """
    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    # Define color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot violin plot
    sns.violinplot(
        data=df,
        x=y_variable,
        y=x_variable,
        palette=color_palette,
        inner=None,  # Remove inner boxplot to avoid clutter
        linewidth=1.5,
        alpha=0.7  # Make the violins slightly transparent
    )

    # Overlay swarm plot
    sns.swarmplot(
        data=df,
        x=y_variable,
        y=x_variable,
        palette=color_palette,
        size=5,  # Size of individual points
        edgecolor="gray",
        linewidth=0.5,
        alpha=0.9
    )

    # Add labels and title
    plt.xlabel(y_variable if y_label is None else y_label, fontsize=14)
    plt.ylabel(x_variable if x_label is None else x_label, fontsize=14)
    plt.title("Combined Violin and Swarm Plot", fontsize=16)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()

# %
def swarm_plot(save_path, df, x_variable, y_variable, x_label=None, y_label=None, color_palette=None):
    """
    Create a swarm plot with dynamically adjusted point sizes to ensure visibility of at least 90% points.
    """
    # Set the seaborn theme to remove top and right spines
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Set the global font size and context for better visualization
    sns.set_context("talk", font_scale=1.2)

    # Define color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Calculate sample sizes for each group in y_variable
    site_counts = df[y_variable].value_counts()

    # Define base marker size
    base_size = 5
    size_mapping = site_counts.apply(lambda n: base_size * np.sqrt(60 / n) if n > 60 else base_size)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the plot for each group to dynamically adjust sizes
    for site in df[y_variable].unique():
        subset = df[df[y_variable] == site]
        point_size = size_mapping[site]

        try:
            # Use swarmplot for most sites
            sns.swarmplot(
                data=subset, x=y_variable, y=x_variable,
                palette=[color_palette[df[y_variable].unique().tolist().index(site)]],
                size=point_size, ax=ax, dodge=True
            )
        except ValueError:
            # If swarmplot fails, fallback to stripplot with jitter for tight spaces
            sns.stripplot(
                data=subset, x=y_variable, y=x_variable,
                palette=[color_palette[df[y_variable].unique().tolist().index(site)]],
                size=point_size, ax=ax, jitter=True, dodge=True
            )

    # Labeling
    ax.set_title('Age distribution', fontsize=20)
    ax.set_xlabel(x_label if x_label else y_variable, fontsize=18)
    ax.set_ylabel(y_label if y_label else x_variable, fontsize=18)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save as svg
    plt.savefig(save_path + f'{x_variable}_swarm_plot.svg', format='svg')
    # Display the plot
    plt.show()
    plt.close()

# def strip_plot(save_path, df, x_variable, y_variable, x_label=None, y_label=None, color_palette=None):
#     """
#     Create a strip plot for the dataset with counts displayed above each column.
#
#     Parameters:
#     - save_path: Path to save the plot.
#     - df: DataFrame containing the data.
#     - x_variable: The variable for the y-axis.
#     - y_variable: The variable for the x-axis.
#     - x_label, y_label: Optional axis labels.
#     - color_palette: Optional color palette for the plot.
#     """
#     # -----------------------------
#     # Local font-size constants
#     # -----------------------------
#     FS_TICK = 10
#     FS_LABEL = 12
#     FS_TITLE = 14
#
#     # Set the seaborn theme to remove top and right spines
#     custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#     sns.set_theme(style="ticks", rc=custom_params)
#
#     # Define color palette
#     if color_palette is None:
#         color_palette = sns.color_palette("husl", len(df[y_variable].unique()))
#
#     # Create the figure and axis
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # Create the strip plot with jitter to reduce overlap
#     sns.stripplot(
#         data=df,
#         x=y_variable,
#         y=x_variable,
#         palette=color_palette,
#         jitter=True,  # Adds horizontal jitter for better point visibility
#         size=6,  # Point size
#         alpha=0.8,  # Slight transparency for better visibility
#         ax=ax
#     )
#
#     # Add counts (n=xxx) above each category
#     category_counts = df[y_variable].value_counts()
#     for i, category in enumerate(df[y_variable].unique()):
#         count = category_counts[category]  # Get the count for the category
#         ax.text(i, df[x_variable].max() + (df[x_variable].max() * 0.05),  # Position slightly above the max value
#                 f'n={count}', ha='center', va='bottom', fontsize=16, color='black')
#
#     # Labeling
#     # ax.set_title('Age distribution', fontsize=20)  # Set title font size
#     ax.set_title(f'{y_label} distribution', fontsize=20, pad=30)
#     ax.set_xlabel(x_label if x_label else x_variable, fontsize=18)  # Set x-axis label font size
#     ax.set_ylabel(y_label if y_label else y_variable, fontsize=18)  # Set y-axis label font size
#
#     # Rotate x-axis labels for better readability
#     ax.tick_params(axis='x', rotation=45, labelsize=16)  # Set x-axis tick label size
#     ax.tick_params(axis='y', labelsize=16)  # Set y-axis tick label size
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#     # Save as svg
#     plt.savefig(save_path + f'{x_variable}_strip_plot.svg', format='svg')
#     # Display the plot
#     plt.show()
#     plt.close()
def strip_plot(save_path, df, x_variable, y_variable, x_label=None, y_label=None, color_palette=None):
    """
    Create a strip plot for the dataset with counts displayed above each column.

    Parameters:
    - save_path: Path to save the plot.
    - df: DataFrame containing the data.
    - x_variable: The numerical variable to plot (on y-axis).
    - y_variable: The categorical variable (on x-axis).
    - x_label, y_label: Optional axis labels.
    - color_palette: Optional color palette for the plot.
    """
    # -----------------------------
    # Local font-size constants
    # -----------------------------
    FS_TICK = 10
    FS_LABEL = 12
    FS_TITLE = 14

    # Theme: remove top/right spines
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Figure size scaled to categories (smaller fonts => smaller figure)
    n_cat = df[y_variable].nunique()
    fig_w = max(6.0, 0.9 * n_cat + 3.0)  # inches
    fig_h = 4.5  # inches
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Strip plot
    sns.stripplot(
        data=df,
        x=y_variable,
        y=x_variable,
        palette=color_palette,
        jitter=True,
        size=6,
        alpha=0.8,
        ax=ax,
    )

    # Counts above each category
    category_counts = df[y_variable].value_counts()
    y_min = df[x_variable].min()
    y_max = df[x_variable].max()
    y_pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0

    for i, category in enumerate(df[y_variable].unique()):
        count = category_counts[category]
        ax.text(
            i,
            y_max + y_pad,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=FS_TICK,
            color="black",
        )

    # Labels / title
    ax.set_title(f"{(x_label if x_label else x_variable)} distribution", fontsize=FS_TITLE, pad=16)
    ax.set_xlabel(y_label if y_label else y_variable, fontsize=FS_LABEL)
    ax.set_ylabel(x_label if x_label else x_variable, fontsize=FS_LABEL)

    ax.tick_params(axis="x", rotation=45, labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)

    plt.tight_layout()
    plt.savefig(save_path + f"{x_variable}_strip_plot.svg", format="svg")
    plt.show()
    plt.close()


# %%
"""
SHIP-Trend
"""
df_SHIP_cleaned = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed_all_cleaned.csv')
df_SHIP_renamed = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP_raw = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/SHIP_Trend_ENIGMA_Sleep_and_cognitive_performance.csv')

df_SHIP_renamed_cld = df_SHIP_renamed.dropna(subset=['Stroop_Test', 'Memory_Test'])

# describe the 'Stroop_Test' and 'Memory_Test' columns
df_SHIP_renamed_cld['Memory_Test'] = df_SHIP_renamed_cld['Memory_Test'].apply(lambda x: x / 16) * 100

# description of df_SHIP_renamed_cld
describe_and_count(df_SHIP_renamed_cld)

# plot the distribution of 'Stroop_Test' and 'Memory_Test'
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df_SHIP_renamed_cld['Stroop_Test'], ax=ax[0])
sns.histplot(df_SHIP_renamed_cld['Memory_Test'], ax=ax[1])
# add title of SHIP-Trend
plt.suptitle('SHIP-Trend')
# add sub-title of 'Stroop_Test' and 'Memory_Test'
ax[0].set_title('Stroop_Test')
ax[1].set_title('Memory_Test')
plt.tight_layout()
plt.show()
plt.close()

# %%
# save the df_SHIP_renamed_cld
# df_SHIP_renamed_cld.to_csv(data_save_path + 'SHIP_Trend_dataset_renamed_target_cleaned.csv', index=False)

# %%
"""
Liege
"""
df_Liege_cleaned = pd.read_csv(data_save_path + 'Liege_dataset_renamed_cleaned.csv')
df_Liege_COF_cleaned = pd.read_csv(data_save_path + 'Liege_COF_dataset_renamed_cleaned.csv')
df_Liege_COGNAP_cleaned = pd.read_csv(data_save_path + 'Liege_COGNAP_dataset_renamed_cleaned.csv')

df_Liege_renamed = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')
df_Liege_COF_renamed = pd.read_csv(data_save_path + 'Liege_COF_dataset_renamed.csv')
df_Liege_COGNAP_renamed = pd.read_csv(data_save_path + 'Liege_COGNAP_dataset_renamed.csv')

df_Liege_renamed_cld = df_Liege_renamed.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_Liege_COF_renamed_cld = df_Liege_COF_renamed.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_Liege_COGNAP_renamed_cld = df_Liege_COGNAP_renamed.dropna(subset=['Stroop_Test', 'Memory_Test'])

df_Liege_renamed_cld['Memory_Test'] = df_Liege_renamed_cld['Memory_Test'] * 100
df_Liege_COF_renamed_cld['Memory_Test'] = df_Liege_COF_renamed_cld['Memory_Test'] * 100
df_Liege_COGNAP_renamed_cld['Memory_Test'] = df_Liege_COGNAP_renamed_cld['Memory_Test'] * 100

sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']

df_Liege_renamed_cld = utils.convert_units(df_Liege_renamed_cld, sleep_dur_cols, sleep_eff_cols)
df_Liege_COF_renamed_cld = utils.convert_units(df_Liege_COF_renamed_cld, sleep_dur_cols, sleep_eff_cols)
df_Liege_COGNAP_renamed_cld = utils.convert_units(df_Liege_COGNAP_renamed_cld, sleep_dur_cols, sleep_eff_cols)

# description of df_Liege_COF_renamed_cld, df_Liege_COGNAP_renamed_cld
print("Liege")
describe_and_count(df_Liege_renamed_cld)

print("Liege_COF")
describe_and_count(df_Liege_COF_renamed_cld)

print("Liege_COGNAP")
describe_and_count(df_Liege_COGNAP_renamed_cld)

# plot the distribution of 'Stroop_Test' and 'Memory_Test'
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df_Liege_renamed_cld['Stroop_Test'], ax=ax[0])
sns.histplot(df_Liege_renamed_cld['Memory_Test'], ax=ax[1])
# add title of Liege
plt.suptitle('Liege')
# add sub-title of 'Stroop_Test' and 'Memory_Test'
ax[0].set_title('Stroop_Test')
ax[1].set_title('Memory_Test')
plt.tight_layout()
plt.show()

# %%
# df_Liege_renamed_cld.to_csv(data_save_path + 'Liege_dataset_renamed_target_cleaned.csv', index=False)

# %%
"""
Karolinska
"""
df_KI_cleaned = pd.read_csv(data_save_path + 'KI_dataset_renamed_cleaned.csv')
df_KI_renamed = pd.read_csv(data_save_path + 'KI_dataset_renamed.csv')

df_KI_renamed_cld = df_KI_renamed.dropna(subset=['Memory_Test'])

df_KI_renamed_cld = utils.convert_units(df_KI_renamed_cld, sleep_dur_cols, sleep_eff_cols)

# description of the df_KI_renamed_cld
describe_and_count(df_KI_renamed_cld)

# for df_KI_demo, rename Depression_HADS to Depression_score
df_KI_demo = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/ENIGMA-Sleep&Cognition_Phenotypic_v4__HB_Jul06.csv')
df_KI_demo.rename(columns={'Depression_HADS': 'Depression_score'}, inplace=True)

df_KI_demo_cld = df_KI_demo.dropna(subset=['Memory_Test'])

df_KI_demo_cld = utils.convert_units(df_KI_demo_cld, sleep_dur_cols, sleep_eff_cols)

describe_and_count(df_KI_demo_cld)

# plot the distribution of 'Memory_Test'
sns.histplot(df_KI_renamed_cld['Memory_Test'])
plt.title('Karolinska')
plt.tight_layout()
plt.show()
plt.close()

# %%
# df_KI_renamed_cld.to_csv(data_save_path + 'KI_dataset_renamed_target_cleaned.csv', index=False)

# %%
"""
Pittsburgh
"""
# df_Pitts_cleaned = pd.read_csv(data_save_path + 'Pitts_dataset_renamed_cleaned.csv')
df_Pitts_renamed = pd.read_csv(data_save_path + 'Pitts_dataset_renamed.csv')
df_Pitts_target_trans = pd.read_csv(data_save_path + 'Pitts_dataset_renamed_target_transformed.csv')

df_Pitts_renamed_cld = df_Pitts_renamed.dropna(subset=['Executive_Functioning', 'Memory_Test1', 'Memory_Test2'])
df_Pitts_target_trans_cld = df_Pitts_target_trans.dropna(subset=['Executive_Functioning', 'Memory_Test1', 'Memory_Test2'])

# describe the 'Executive_Functioning', 'Memory_Test1', 'Memory_Test2' columns
print(df_Pitts_renamed_cld[['Executive_Functioning', 'Memory_Test1', 'Memory_Test2']].describe())
print(df_Pitts_target_trans_cld[['Executive_Functioning', 'Memory_Test1', 'Memory_Test2']].describe())

# add a column 'APOE4' to df_Pitts_renamed_cld. All subject is missing value.
df_Pitts_renamed_cld['APOE4'] = None
# description of df_Pitts_renamed_cld

df_Pitts_renamed_cld = utils.convert_units(df_Pitts_renamed_cld, sleep_dur_cols, sleep_eff_cols)
describe_and_count(df_Pitts_renamed_cld)

# plot the distribution of 'Executive_Functioning', 'Memory_Test1', 'Memory_Test2'
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.histplot(df_Pitts_target_trans_cld['Executive_Functioning'], ax=ax[0])
sns.histplot(df_Pitts_target_trans_cld['Memory_Test1'], ax=ax[1])
sns.histplot(df_Pitts_target_trans_cld['Memory_Test2'], ax=ax[2])
# add title of Pittsburgh
plt.suptitle('Pittsburgh')
# add sub-title of 'Executive_Functioning', 'Memory_Test1', 'Memory_Test2'
ax[0].set_title('Executive_Functioning')
ax[1].set_title('Memory_Test1')
ax[2].set_title('Memory_Test2')
plt.tight_layout()
plt.show()
plt.close()

# %%
# df_Pitts_target_trans_cld.to_csv(data_save_path + 'Pitts_dataset_renamed_target_cleaned.csv', index=False)

# %%
"""
Juelich
"""
df_Juelich_SomnoSafe_1 = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_1.csv', index_col=0)
df_Juelich_SomnoSafe_2 = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_2.csv', index_col=0)
df_Juelich_SomnoSafe_SD = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_SD.csv', index_col=0)

df_Juelich_SomnoSafe_1 = utils.convert_units(df_Juelich_SomnoSafe_1, sleep_dur_cols, sleep_eff_cols)
df_Juelich_SomnoSafe_2 = utils.convert_units(df_Juelich_SomnoSafe_2, sleep_dur_cols, sleep_eff_cols)
df_Juelich_SomnoSafe_SD = utils.convert_units(df_Juelich_SomnoSafe_SD, sleep_dur_cols, sleep_eff_cols)

describe_and_count(df_Juelich_SomnoSafe_1)
describe_and_count(df_Juelich_SomnoSafe_2)
describe_and_count(df_Juelich_SomnoSafe_SD)
# for Letter_Sensitivity, Spatial_Sensitivity in df_Juelich_SomnoSafe_1, df_Juelich_SomnoSafe_2, and df_Juelich_SomnoSafe_SD, value * 100
df_Juelich_SomnoSafe_1['Letter_Sensitivity'] = df_Juelich_SomnoSafe_1['Letter_Sensitivity'] * 100
df_Juelich_SomnoSafe_2['Letter_Sensitivity'] = df_Juelich_SomnoSafe_2['Letter_Sensitivity'] * 100
df_Juelich_SomnoSafe_SD['Letter_Sensitivity'] = df_Juelich_SomnoSafe_SD['Letter_Sensitivity'] * 100
df_Juelich_SomnoSafe_1['Spatial_Sensitivity'] = df_Juelich_SomnoSafe_1['Spatial_Sensitivity'] * 100
df_Juelich_SomnoSafe_2['Spatial_Sensitivity'] = df_Juelich_SomnoSafe_2['Spatial_Sensitivity'] * 100
df_Juelich_SomnoSafe_SD['Spatial_Sensitivity'] = df_Juelich_SomnoSafe_SD['Spatial_Sensitivity'] * 100

# plot the distribution of 'PVT_reaction_speed' of df_Juelich_SomnoSafe_1, df_Juelich_SomnoSafe_2, and df_Juelich_SomnoSafe_SD
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.histplot(df_Juelich_SomnoSafe_1['PVT_reaction_speed'], ax=ax[0])
sns.histplot(df_Juelich_SomnoSafe_2['PVT_reaction_speed'], ax=ax[1])
sns.histplot(df_Juelich_SomnoSafe_SD['PVT_reaction_speed'], ax=ax[2])
# add title of Juelich
plt.suptitle('Juelich')
# add sub-title of 'PVT_reaction_speed'
ax[0].set_title('PVT_reaction_speed')
ax[1].set_title('PVT_reaction_speed')
ax[2].set_title('PVT_reaction_speed')
plt.tight_layout()
plt.show()
plt.close()

# plot the distribution of 'Letter_HR', 'Letter_Rtmean', 'Letter_Sensitivity', 'Spatial_HR', 'Spatial_Rtmean', 'Spatial_Sensitivity'
# first row is 'Letter', second row is 'Spatial'
# for df_Juelich_SomnoSafe_1, df_Juelich_SomnoSafe_2, and df_Juelich_SomnoSafe_SD
fig, ax = plt.subplots(6, 3, figsize=(18, 18))
sns.histplot(df_Juelich_SomnoSafe_1['Letter_HR'], ax=ax[0, 0])
sns.histplot(df_Juelich_SomnoSafe_1['Letter_Rtmean'], ax=ax[0, 1])
sns.histplot(df_Juelich_SomnoSafe_1['Letter_Sensitivity'], ax=ax[0, 2])
sns.histplot(df_Juelich_SomnoSafe_1['Spatial_HR'], ax=ax[1, 0])
sns.histplot(df_Juelich_SomnoSafe_1['Spatial_Rtmean'], ax=ax[1, 1])
sns.histplot(df_Juelich_SomnoSafe_1['Spatial_Sensitivity'], ax=ax[1, 2])
sns.histplot(df_Juelich_SomnoSafe_2['Letter_HR'], ax=ax[2, 0])
sns.histplot(df_Juelich_SomnoSafe_2['Letter_Rtmean'], ax=ax[2, 1])
sns.histplot(df_Juelich_SomnoSafe_2['Letter_Sensitivity'], ax=ax[2, 2])
sns.histplot(df_Juelich_SomnoSafe_2['Spatial_HR'], ax=ax[3, 0])
sns.histplot(df_Juelich_SomnoSafe_2['Spatial_Rtmean'], ax=ax[3, 1])
sns.histplot(df_Juelich_SomnoSafe_2['Spatial_Sensitivity'], ax=ax[3, 2])
sns.histplot(df_Juelich_SomnoSafe_SD['Letter_HR'], ax=ax[4, 0])
sns.histplot(df_Juelich_SomnoSafe_SD['Letter_Rtmean'], ax=ax[4, 1])
sns.histplot(df_Juelich_SomnoSafe_SD['Letter_Sensitivity'], ax=ax[4, 2])
sns.histplot(df_Juelich_SomnoSafe_SD['Spatial_HR'], ax=ax[5, 0])
sns.histplot(df_Juelich_SomnoSafe_SD['Spatial_Rtmean'], ax=ax[5, 1])
sns.histplot(df_Juelich_SomnoSafe_SD['Spatial_Sensitivity'], ax=ax[5, 2])
# add title of Juelich
plt.suptitle('Juelich')
# add sub-title of 'Letter_HR', 'Letter_Rtmean', 'Letter_Sensitivity', 'Spatial_HR', 'Spatial_Rtmean', 'Spatial_Sensitivity'
ax[0, 0].set_title('Letter_HR')
ax[0, 1].set_title('Letter_Rtmean')
ax[0, 2].set_title('Letter_Sensitivity')
ax[1, 0].set_title('Spatial_HR')
ax[1, 1].set_title('Spatial_Rtmean')
ax[1, 2].set_title('Spatial_Sensitivity')
ax[2, 0].set_title('Letter_HR')
ax[2, 1].set_title('Letter_Rtmean')
ax[2, 2].set_title('Letter_Sensitivity')
ax[3, 0].set_title('Spatial_HR')
ax[3, 1].set_title('Spatial_Rtmean')
ax[3, 2].set_title('Spatial_Sensitivity')
ax[4, 0].set_title('Letter_HR')
ax[4, 1].set_title('Letter_Rtmean')
ax[4, 2].set_title('Letter_Sensitivity')
ax[5, 0].set_title('Spatial_HR')
ax[5, 1].set_title('Spatial_Rtmean')
ax[5, 2].set_title('Spatial_Sensitivity')
plt.tight_layout()
plt.show()
plt.close()

# %%
# df_Juelich_SomnoSafe_1.to_csv(data_save_path + 'Juelich_SomnoSafe_1_target_cleaned.csv')
# df_Juelich_SomnoSafe_2.to_csv(data_save_path + 'Juelich_SomnoSafe_2_target_cleaned.csv')
# df_Juelich_SomnoSafe_SD.to_csv(data_save_path + 'Juelich_SomnoSafe_SD_target_cleaned.csv')

# %%
"""
Rotterdam

TODO: The saving of ture target value, predicted target value and the age is not correct. Revise it in the apply model script.
"""
df_EMC_results = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/XGBoost/EMC/Stroop/Brain/results_values.csv')

# print the description of df_EMC_results
print(df_EMC_results.describe())

# %%
EMC_True = df_EMC_results['EMC_true'].values

# plot the distribution of 'EMC_True'
sns.histplot(EMC_True)
plt.title('Rotterdam')
plt.tight_layout()
plt.show()
plt.close()

# %%
"""
San Diego
"""
df_VETSA_renamed = pd.read_csv(data_save_path + 'VETSA_dataset_renamed.csv', index_col=0)

df_VETSA_renamed_cld = df_VETSA_renamed.dropna(subset=['Stroop_Test', 'Memory_Test_Digit', 'Memory_Test_Letter'])

# df_VETSA_renamed_cld = utils.convert_units(df_VETSA_renamed_cld, sleep_dur_cols, sleep_eff_cols)

# description of df_VETSA_renamed_cld
describe_and_count(df_VETSA_renamed_cld)

# for Memory_Test_Digit, Memory_Test_Letter in df_VETSA_renamed_cld, value * 100
df_VETSA_renamed_cld['Memory_Test_Digit'] = df_VETSA_renamed_cld['Memory_Test_Digit'] * 100
df_VETSA_renamed_cld['Memory_Test_Letter'] = df_VETSA_renamed_cld['Memory_Test_Letter'] * 100

# plot the distribution of 'Stroop_Test', 'Memory_Test_Digit', 'Memory_Test_Letter'
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.histplot(df_VETSA_renamed_cld['Stroop_Test'], ax=ax[0])
sns.histplot(df_VETSA_renamed_cld['Memory_Test_Digit'], ax=ax[1])
sns.histplot(df_VETSA_renamed_cld['Memory_Test_Letter'], ax=ax[2])
# add title of San Diego
plt.suptitle('San Diego')
# add sub-title of 'Stroop_Test', 'Memory_Test_Digit', 'Memory_Test_Letter'
ax[0].set_title('Stroop_Test')
ax[1].set_title('Memory_Test_Digit')
ax[2].set_title('Memory_Test_Letter')
plt.tight_layout()
plt.show()
plt.close()

# %%
# df_VETSA_renamed_cld.to_csv(data_save_path + 'VETSA_dataset_renamed_target_cleaned.csv', index=False)

# %%
# make dataframe
df_SHIP_age = df_SHIP_renamed_cld[['Age_at_Scan']]
df_SHIP_age['Dataset'] = 'Greifswald'
df_Liege_age = df_Liege_renamed_cld[['Age_at_Scan']]
df_Liege_age['Dataset'] = 'Liège'
df_KI_age = df_KI_renamed_cld[['Age_at_Scan']]
df_KI_age['Dataset'] = 'Stockholm'
df_Pitts_age = df_Pitts_target_trans_cld[['Age_at_Scan']]
df_Pitts_age['Dataset'] = 'Pittsburgh'
df_Juelich_age = df_Juelich_SomnoSafe_1[['Age_at_Scan']]
df_Juelich_age['Dataset'] = 'Jülich'
df_EMC_age = df_EMC_results[['Age_at_Scan']]
df_EMC_age['Dataset'] = 'Rotterdam'
df_VETSA_age = df_VETSA_renamed_cld[['Age_at_Scan']]
df_VETSA_age['Dataset'] = 'San Diego'

df_age = pd.concat([df_SHIP_age, df_EMC_age, df_VETSA_age, df_Liege_age, df_KI_age, df_Pitts_age, df_Juelich_age])

# %%
# description of df_age
print(df_age.describe())

# %%
"""
SHIP-Trend: #18476A
EMC: #F3A3A7
VETSA: #E4BDE5
Liege: #56BBB8
KI: PASTEL GREEN #7EA8D3
Pitts: STONE #96D3A1
Juelich: #90A2A5
"""
custom_colors = ['#18476A', '#F3A3A7', '#E4BDE5', '#56BBB8', '#7EA8D3', '#96D3A1', '#90A2A5']
reverse_custom_colors = custom_colors[::-1]

# %%
ridge_plot_new(result_save_path, df_age, 'Age_at_Scan', 'Dataset', x_label='Age', color_palette=reverse_custom_colors)
# strip_plot(result_save_path, df_age, 'Age_at_Scan', 'Dataset', 'Dataset', 'Age', custom_colors)

# %%
"""
Make correlation matrix plot of 'Stroop_Test', 'Memory_Test' from each site
"""
Stroop_SHIP = df_SHIP_renamed_cld[['Stroop_Test', 'Age_at_Scan']]
Stroop_SHIP['Dataset'] = 'Greifswald'
Memory_SHIP = df_SHIP_renamed_cld[['Memory_Test', 'Age_at_Scan']]
Memory_SHIP['Dataset'] = 'Greifswald'

Stroop_Liege = df_Liege_renamed_cld[['Stroop_Test', 'Age_at_Scan']]
Stroop_Liege['Dataset'] = 'Liège'
Memory_Liege = df_Liege_renamed_cld[['Memory_Test', 'Age_at_Scan']]
Memory_Liege['Dataset'] = 'Liège'

Memory_KI = df_KI_renamed_cld[['Memory_Test', 'Age_at_Scan']]
Memory_KI['Dataset'] = 'Stockholm'

Stroop_Pitts = df_Pitts_target_trans_cld[['Executive_Functioning', 'Age_at_Scan']]
Stroop_Pitts['Dataset'] = 'Pittsburgh'
Memory_Pitts_1 = df_Pitts_target_trans_cld[['Memory_Test1', 'Age_at_Scan']]
Memory_Pitts_1['Dataset'] = 'Pittsburgh'
Memory_Pitts_2 = df_Pitts_target_trans_cld[['Memory_Test2', 'Age_at_Scan']]
Memory_Pitts_2['Dataset'] = 'Pittsburgh'

Memory_Juelich_Letter_HR_1 = df_Juelich_SomnoSafe_1[['Letter_HR']]
Memory_Juelich_Letter_HR_1['Age_at_Scan'] = df_Juelich_SomnoSafe_1[['Age_at_Scan']]
Memory_Juelich_Letter_HR_1['Dataset'] = 'Jülich'

Memory_Juelich_Letter_Rtmean_1 = df_Juelich_SomnoSafe_1[['Letter_Rtmean']]
Memory_Juelich_Letter_Rtmean_1['Age_at_Scan'] = df_Juelich_SomnoSafe_1[['Age_at_Scan']]
Memory_Juelich_Letter_Rtmean_1['Dataset'] = 'Jülich'

Memory_Juelich_Letter_Sensitivity_1 = df_Juelich_SomnoSafe_1[['Letter_Sensitivity']]
Memory_Juelich_Letter_Sensitivity_1['Age_at_Scan'] = df_Juelich_SomnoSafe_1[['Age_at_Scan']]
Memory_Juelich_Letter_Sensitivity_1['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_HR_1 = df_Juelich_SomnoSafe_1[['Spatial_HR']]
Memory_Juelich_Spatial_HR_1['Age_at_Scan'] = df_Juelich_SomnoSafe_1[['Age_at_Scan']]
Memory_Juelich_Spatial_HR_1['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_Rtmean_1 = df_Juelich_SomnoSafe_1[['Spatial_Rtmean']]
Memory_Juelich_Spatial_Rtmean_1['Age_at_Scan'] = df_Juelich_SomnoSafe_1[['Age_at_Scan']]
Memory_Juelich_Spatial_Rtmean_1['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_Sensitivity_1 = df_Juelich_SomnoSafe_1[['Spatial_Sensitivity']]
Memory_Juelich_Spatial_Sensitivity_1['Age_at_Scan'] = df_Juelich_SomnoSafe_1[['Age_at_Scan']]
Memory_Juelich_Spatial_Sensitivity_1['Dataset'] = 'Jülich'

Memory_Juelich_Letter_HR_2 = df_Juelich_SomnoSafe_2[['Letter_HR']]
Memory_Juelich_Letter_HR_2['Age_at_Scan'] = df_Juelich_SomnoSafe_2[['Age_at_Scan']]
Memory_Juelich_Letter_HR_2['Dataset'] = 'Jülich'

Memory_Juelich_Letter_Rtmean_2 = df_Juelich_SomnoSafe_2[['Letter_Rtmean']]
Memory_Juelich_Letter_Rtmean_2['Age_at_Scan'] = df_Juelich_SomnoSafe_2[['Age_at_Scan']]
Memory_Juelich_Letter_Rtmean_2['Dataset'] = 'Jülich'

Memory_Juelich_Letter_Sensitivity_2 = df_Juelich_SomnoSafe_2[['Letter_Sensitivity']]
Memory_Juelich_Letter_Sensitivity_2['Age_at_Scan'] = df_Juelich_SomnoSafe_2[['Age_at_Scan']]
Memory_Juelich_Letter_Sensitivity_2['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_HR_2 = df_Juelich_SomnoSafe_2[['Spatial_HR']]
Memory_Juelich_Spatial_HR_2['Age_at_Scan'] = df_Juelich_SomnoSafe_2[['Age_at_Scan']]
Memory_Juelich_Spatial_HR_2['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_Rtmean_2 = df_Juelich_SomnoSafe_2[['Spatial_Rtmean']]
Memory_Juelich_Spatial_Rtmean_2['Age_at_Scan'] = df_Juelich_SomnoSafe_2[['Age_at_Scan']]
Memory_Juelich_Spatial_Rtmean_2['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_Sensitivity_2 = df_Juelich_SomnoSafe_2[['Spatial_Sensitivity']]
Memory_Juelich_Spatial_Sensitivity_2['Age_at_Scan'] = df_Juelich_SomnoSafe_2[['Age_at_Scan']]
Memory_Juelich_Spatial_Sensitivity_2['Dataset'] = 'Jülich'

Memory_Juelich_Letter_HR_SD = df_Juelich_SomnoSafe_SD[['Letter_HR']]
Memory_Juelich_Letter_HR_SD['Age_at_Scan'] = df_Juelich_SomnoSafe_SD[['Age_at_Scan']]
Memory_Juelich_Letter_HR_SD['Dataset'] = 'Jülich'

Memory_Juelich_Letter_Rtmean_SD = df_Juelich_SomnoSafe_SD[['Letter_Rtmean']]
Memory_Juelich_Letter_Rtmean_SD['Age_at_Scan'] = df_Juelich_SomnoSafe_SD[['Age_at_Scan']]
Memory_Juelich_Letter_Rtmean_SD['Dataset'] = 'Jülich'

Memory_Juelich_Letter_Sensitivity_SD = df_Juelich_SomnoSafe_SD[['Letter_Sensitivity']]
Memory_Juelich_Letter_Sensitivity_SD['Age_at_Scan'] = df_Juelich_SomnoSafe_SD[['Age_at_Scan']]
Memory_Juelich_Letter_Sensitivity_SD['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_HR_SD = df_Juelich_SomnoSafe_SD[['Spatial_HR']]
Memory_Juelich_Spatial_HR_SD['Age_at_Scan'] = df_Juelich_SomnoSafe_SD[['Age_at_Scan']]
Memory_Juelich_Spatial_HR_SD['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_Rtmean_SD = df_Juelich_SomnoSafe_SD[['Spatial_Rtmean']]
Memory_Juelich_Spatial_Rtmean_SD['Age_at_Scan'] = df_Juelich_SomnoSafe_SD[['Age_at_Scan']]
Memory_Juelich_Spatial_Rtmean_SD['Dataset'] = 'Jülich'

Memory_Juelich_Spatial_Sensitivity_SD = df_Juelich_SomnoSafe_SD[['Spatial_Sensitivity']]
Memory_Juelich_Spatial_Sensitivity_SD['Age_at_Scan'] = df_Juelich_SomnoSafe_SD[['Age_at_Scan']]
Memory_Juelich_Spatial_Sensitivity_SD['Dataset'] = 'Jülich'

Stroop_EMC = pd.DataFrame({'Stroop_Test': EMC_True})
Stroop_EMC['Age_at_Scan'] = df_EMC_results[['Age_at_Scan']]
Stroop_EMC['Dataset'] = 'Rotterdam'

Stroop_VETSA = df_VETSA_renamed_cld[['Stroop_Test', 'Age_at_Scan']]
Stroop_VETSA['Dataset'] = 'San Diego'
Memory_VETSA_Digit = df_VETSA_renamed_cld[['Memory_Test_Digit']]
Memory_VETSA_Digit['Age_at_Scan'] = df_VETSA_renamed_cld[['Age_at_Scan']]
Memory_VETSA_Digit['Dataset'] = 'San Diego'
Memory_VETSA_Letter = df_VETSA_renamed_cld[['Memory_Test_Letter']]
Memory_VETSA_Letter['Age_at_Scan'] = df_VETSA_renamed_cld[['Age_at_Scan']]
Memory_VETSA_Letter['Dataset'] = 'San Diego'

# %%
"""
SHIP-Trend: #18476A
EMC: #F3A3A7
VETSA: #E4BDE5
Liege: #56BBB8
KI: PASTEL GREEN #7EA8D3
Pitts: STONE #96D3A1
Juelich: #90A2A5
"""
# Strip plot of Stroop test scores across sites
# rename Stroop_Pitts's 'Executive_Functioning' column to 'Stroop_Test'
Stroop_Pitts.rename(columns={'Executive_Functioning': 'Stroop_Test'}, inplace=True)
Stroop_Pitts['Dataset'] = 'Pittsburgh'
df_Stroop = pd.concat([Stroop_SHIP, Stroop_EMC, Stroop_VETSA, Stroop_Liege, Stroop_Pitts])
custom_colors_Stroop = ['#18476A', '#F3A3A7', '#E4BDE5', '#56BBB8', '#96D3A1']

# strip_plot(result_save_path, df_Stroop, 'Stroop_Test', 'Dataset', 'Dataset', 'Stroop test score', custom_colors_Stroop)

# %%
Memory_Pitts_1.rename(columns={'Memory_Test1': 'Memory_Test'}, inplace=True)
Memory_Pitts_1['Dataset'] = 'Pittsburgh (Letter)'  # 'Pittsburgh Letter'
Memory_Pitts_2.rename(columns={'Memory_Test2': 'Memory_Test'}, inplace=True)
Memory_Pitts_2['Dataset'] = 'Pittsburgh (Spatial)'  # 'Pittsburgh Spatial'
Memory_Juelich_Letter_HR_1.rename(columns={'Letter_HR': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_HR_1['Dataset'] = 'Jülich Letter HR'
Memory_Juelich_Letter_Rtmean_1.rename(columns={'Letter_Rtmean': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_Rtmean_1['Dataset'] = 'Jülich Letter Rtmean'
Memory_Juelich_Letter_Sensitivity_1.rename(columns={'Letter_Sensitivity': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_Sensitivity_1['Dataset'] = 'Jülich (Letter)'  # 'Jülich Letter Sensitivity'
Memory_Juelich_Spatial_HR_1.rename(columns={'Spatial_HR': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_HR_1['Dataset'] = 'Jülich Spatial HR'
Memory_Juelich_Spatial_Rtmean_1.rename(columns={'Spatial_Rtmean': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_Rtmean_1['Dataset'] = 'Jülich Spatial Rtmean'
Memory_Juelich_Spatial_Sensitivity_1.rename(columns={'Spatial_Sensitivity': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_Sensitivity_1['Dataset'] = 'Jülich (Spatial)'  # 'Jülich Spatial Sensitivity'
Memory_Juelich_Letter_HR_2.rename(columns={'Letter_HR': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_HR_2['Dataset'] = 'Jülich Letter HR'
Memory_Juelich_Letter_Rtmean_2.rename(columns={'Letter_Rtmean': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_Rtmean_2['Dataset'] = 'Jülich Letter Rtmean'
Memory_Juelich_Letter_Sensitivity_2.rename(columns={'Letter_Sensitivity': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_Sensitivity_2['Dataset'] = 'Jülich Letter'  # 'Jülich Letter Sensitivity'
Memory_Juelich_Spatial_HR_2.rename(columns={'Spatial_HR': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_HR_2['Dataset'] = 'Jülich Spatial HR'
Memory_Juelich_Spatial_Rtmean_2.rename(columns={'Spatial_Rtmean': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_Rtmean_2['Dataset'] = 'Jülich Spatial Rtmean'
Memory_Juelich_Spatial_Sensitivity_2.rename(columns={'Spatial_Sensitivity': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_Sensitivity_2['Dataset'] = 'Jülich Spatial'  # 'Jülich Spatial Sensitivity'
Memory_Juelich_Letter_HR_SD.rename(columns={'Letter_HR': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_HR_SD['Dataset'] = 'Jülich Letter HR'
Memory_Juelich_Letter_Rtmean_SD.rename(columns={'Letter_Rtmean': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_Rtmean_SD['Dataset'] = 'Jülich Letter Rtmean'
Memory_Juelich_Letter_Sensitivity_SD.rename(columns={'Letter_Sensitivity': 'Memory_Test'}, inplace=True)
Memory_Juelich_Letter_Sensitivity_SD['Dataset'] = 'Jülich Letter Sensitivity'
Memory_Juelich_Spatial_HR_SD.rename(columns={'Spatial_HR': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_HR_SD['Dataset'] = 'Jülich Spatial HR'
Memory_Juelich_Spatial_Rtmean_SD.rename(columns={'Spatial_Rtmean': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_Rtmean_SD['Dataset'] = 'Jülich Spatial Rtmean'
Memory_Juelich_Spatial_Sensitivity_SD.rename(columns={'Spatial_Sensitivity': 'Memory_Test'}, inplace=True)
Memory_Juelich_Spatial_Sensitivity_SD['Dataset'] = 'Jülich Spatial Sensitivity'
Memory_VETSA_Digit.rename(columns={'Memory_Test_Digit': 'Memory_Test'}, inplace=True)
Memory_VETSA_Digit['Dataset'] = 'San Diego (Digit)'  # 'San Diego Digit'
Memory_VETSA_Letter.rename(columns={'Memory_Test_Letter': 'Memory_Test'}, inplace=True)
Memory_VETSA_Letter['Dataset'] = 'San Diego (Letter)'  # 'San Diego Letter'

"""
SHIP-Trend: #18476A
EMC: #F3A3A7
VETSA: #E4BDE5
Liege: #56BBB8
KI: #7EA8D3
Pitts: #96D3A1
Juelich: #90A2A5
"""
# Strip plot of Memory test scores across sites
df_Memory_1 = pd.concat([Memory_SHIP,
                         Memory_VETSA_Letter, Memory_VETSA_Digit,
                         Memory_Liege, Memory_KI, Memory_Pitts_1, Memory_Pitts_2,
                         Memory_Juelich_Letter_Sensitivity_1, Memory_Juelich_Spatial_Sensitivity_1])
custom_colors_Memory = ['#18476A', '#E4BDE5', '#E4BDE5', '#56BBB8', '#7EA8D3', '#96D3A1', '#96D3A1',
                        '#90A2A5', '#90A2A5']

# strip_plot(result_save_path, df_Memory_1, 'Memory_Test', 'Dataset', 'Dataset', 'Memory test score', custom_colors_Memory)

# %%
# def strip_plot_subplot(ax, df, x_variable, y_variable, x_label=None, y_label=None, color_palette=None, title=None):
#     """
#     Create a strip plot for the dataset as a subplot with counts displayed above each column.
#
#     Parameters:
#     - ax: The axis object for the subplot.
#     - df: DataFrame containing the data.
#     - x_variable: The variable for the y-axis.
#     - y_variable: The variable for the x-axis.
#     - x_label, y_label: Optional axis labels.
#     - color_palette: Optional color palette for the plot.
#     - title: Optional title for the subplot.
#     """
#     # Define color palette
#     if color_palette is None:
#         color_palette = sns.color_palette("husl", len(df[y_variable].unique()))
#
#     # Create the strip plot with jitter to reduce overlap
#     sns.stripplot(
#         data=df,
#         x=y_variable,
#         y=x_variable,
#         palette=color_palette,
#         jitter=True,  # Adds horizontal jitter for better point visibility
#         size=6,  # Point size
#         alpha=0.8,  # Slight transparency for better visibility
#         ax=ax
#     )
#
#     # Add counts (n=xxx) above each category
#     category_counts = df[y_variable].value_counts()
#     for i, category in enumerate(df[y_variable].unique()):
#         count = category_counts[category]
#         ax.text(i, df[x_variable].max() + (df[x_variable].max() * 0.05),  # Position slightly above the max value
#                 f'n={count}', ha='center', va='bottom', fontsize=12, color='black')
#
#     # Labeling
#     # ax.set_title(title if title else f'{y_label} distribution', fontsize=16)
#     # ax.set_xlabel(x_label if x_label else x_variable, fontsize=14)
#     ax.set_ylabel(y_label if y_label else y_variable, fontsize=14)
#
#     # Rotate x-axis labels for better readability
#     ax.tick_params(axis='x', rotation=30, labelsize=12)
#     ax.tick_params(axis='y', labelsize=12)
#
# def create_strip_plot_subplots(save_path, dfs, x_vars, y_var, x_labels, y_labels, color_palettes):
#     """
#     Create multiple strip plots in a single figure with subplots arranged vertically.
#
#     Parameters:
#     - save_path: Path to save the figure.
#     - dfs: List of DataFrames.
#     - x_vars: List of variables for the y-axis.
#     - y_var: Common variable for the x-axis.
#     - x_labels: List of x-axis labels.
#     - y_labels: List of y-axis labels.
#     - titles: List of subplot titles.
#     - color_palettes: List of color palettes for each subplot.
#     """
#     # Create subplots arranged vertically
#     fig, axes = plt.subplots(len(dfs), 1, figsize=(10, 4 * len(dfs)), sharex=False)
#
#     # # Loop through each dataset and create a subplot
#     # for i, (ax, df, x_var, x_label, y_label, title, palette) in enumerate(zip(axes, dfs, x_vars, x_labels, y_labels, titles, color_palettes)):
#     #     strip_plot_subplot(ax, df, x_var, y_var, x_label, y_label, palette, title)
#
#     # Loop through each dataset and create a subplot
#     for i, (ax, df, x_var, y_label, palette) in enumerate(zip(axes, dfs, x_vars, y_labels, color_palettes)):
#             strip_plot_subplot(ax, df, x_var, y_var, x_label=None, y_label=y_label, color_palette=palette)
#
#             if i < len(axes) - 1:  # every axis except the bottom one
#                 ax.set_xlabel('')  # remove seaborn’s default label
#
#     # add xlabel to the last subplot
#     axes[-1].set_xlabel(x_labels, fontsize=14)
#     # Adjust layout and save the figure
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(save_path + 'strip_plots_subplots_vertical.png', format='png')
#     plt.savefig(save_path + 'strip_plots_subplots_vertical.svg', format='svg')
#     plt.close()

# %%
def strip_plot_subplot(ax, df, x_variable, y_variable,
                      x_label=None, y_label=None, color_palette=None, title=None):
    """
    Strip plot subplot with counts above each category.

    x_variable: numeric (plotted on y-axis)
    y_variable: categorical (plotted on x-axis)
    x_label: label for numeric axis (y-axis)
    y_label: label for categorical axis (x-axis)
    """
    # -----------------------------
    # Local font-size constants
    # -----------------------------
    FS_TICK = 10
    FS_LABEL = 12
    FS_TITLE = 14

    # Theme: remove top/right spines
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Color palette
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(df[y_variable].unique()))

    # Strip plot
    sns.stripplot(
        data=df,
        x=y_variable,
        y=x_variable,
        palette=color_palette,
        jitter=True,
        size=4,
        alpha=0.8,
        ax=ax,
    )

    # Counts above each category (robust padding)
    category_counts = df[y_variable].value_counts()
    y_min = df[x_variable].min()
    y_max = df[x_variable].max()
    y_pad = 0.05 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0

    for i, category in enumerate(df[y_variable].unique()):
        count = category_counts[category]
        ax.text(
            i,
            y_max + y_pad,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=FS_TICK,
            color="black",
        )

    # Title (optional)
    if title is not None:
        ax.set_title(title, fontsize=FS_TITLE, pad=10)

    # Axis labels: y-axis is numeric, x-axis is categorical
    ax.set_ylabel(x_label if x_label else x_variable, fontsize=FS_LABEL)
    if y_label is not None:
        ax.set_xlabel(y_label, fontsize=FS_LABEL)

    # Ticks
    ax.tick_params(axis="x", rotation=45, labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)


def create_strip_plot_subplots(save_path, dfs, x_vars, y_var, x_labels, y_labels, color_palettes):
    """
    Create multiple strip plots vertically.

    dfs: list of DataFrames
    x_vars: list of numeric columns (one per df)
    y_var: categorical column shared across dfs (x-axis categories)
    x_labels: string for bottom x-axis label (e.g., "Dataset")
    y_labels: list of y-axis labels (one per subplot; e.g. ["Age", ...])
    color_palettes: list of palettes (one per subplot)
    """
    # -----------------------------
    # Local font-size constants
    # -----------------------------
    FS_TICK = 10
    FS_LABEL = 12
    FS_TITLE = 14

    # Theme once
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Figure size scaled to categories + rows (matching your single-plot sizing logic)
    n_rows = len(dfs)
    n_cat = max(df[y_var].nunique() for df in dfs)
    fig_w = 6  # inches, max(6.0, 0.9 * n_cat + 3.0)
    fig_h = 8  # inches, max(3.5, 2.8 * n_rows + 0.8)

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for i, (ax, df, x_var, y_lab, palette) in enumerate(zip(axes, dfs, x_vars, y_labels, color_palettes)):
        # y_lab describes the numeric variable -> y-axis label
        strip_plot_subplot(
            ax=ax,
            df=df,
            x_variable=x_var,
            y_variable=y_var,
            x_label=y_lab,
            y_label=None,       # xlabel controlled below
            color_palette=palette,
            title=None,
        )

        # Remove xlabel from all but the last subplot (ticks remain)
        if i < len(axes) - 1:
            ax.set_xlabel("")

    # Bottom xlabel (categorical axis)
    axes[-1].set_xlabel(x_labels, fontsize=FS_LABEL)

    plt.tight_layout()
    # Save first, then show
    plt.savefig(save_path + "strip_plots_subplots_vertical.png", format="png")
    plt.savefig(save_path + "strip_plots_subplots_vertical.svg", format="svg")
    plt.show()
    plt.close()


# %%
# DataFrames and configurations for the subplots
# DataFrames and configurations for the subplots
dfs = [df_age, df_Stroop, df_Memory_1]
x_vars = ['Age_at_Scan', 'Stroop_Test', 'Memory_Test']
y_var = 'Dataset'
# x_labels = ['Age', 'Stroop Test Score', 'Memory Test Score']
# y_labels = ['Dataset', 'Dataset', 'Dataset']
x_labels = 'Dataset'
y_labels = ['Age', 'Stroop Test Score', 'Memory Test Score']
# titles = ['Age Distribution', 'Stroop Test Score Distribution', 'Memory Test Score Distribution']
color_palettes = [custom_colors, custom_colors_Stroop, custom_colors_Memory]

# Call the function to create the subplots
create_strip_plot_subplots(result_save_path, dfs, x_vars, y_var, x_labels, y_labels, color_palettes)

# %%
# sort each Stroop dataframe by 'Age_at_Scan'
# Stroop_SHIP = Stroop_SHIP.sort_values(by='Age_at_Scan').reset_index(drop=True)
# Stroop_Liege = Stroop_Liege.sort_values(by='Age_at_Scan').reset_index(drop=True)
# Stroop_Pitts = Stroop_Pitts.sort_values(by='Age_at_Scan').reset_index(drop=True)
# Stroop_EMC = Stroop_EMC.sort_values(by='Age_at_Scan').reset_index(drop=True)
# Stroop_VETSA = Stroop_VETSA.sort_values(by='Age_at_Scan').reset_index(drop=True)

# avg_corr_Stroop_SHIP_Liege = np.mean(corr_cross_sites(Stroop_SHIP, Stroop_Liege)[0])
# avg_p_value_Stroop_SHIP_Liege = np.mean(corr_cross_sites(Stroop_SHIP, Stroop_Liege)[1])
# print(f'Average correlation of Stroop between SHIP and Liege: {avg_corr_Stroop_SHIP_Liege}', f'Average p-value: {avg_p_value_Stroop_SHIP_Liege}')
#
# avg_corr_Stroop_SHIP_Pitts = np.mean(corr_cross_sites(Stroop_SHIP, Stroop_Pitts)[0])
# avg_p_value_Stroop_SHIP_Pitts = np.mean(corr_cross_sites(Stroop_SHIP, Stroop_Pitts)[1])
# print(f'Average correlation of Stroop between SHIP and Pittsburgh: {avg_corr_Stroop_SHIP_Pitts}', f'Average p-value: {avg_p_value_Stroop_SHIP_Pitts}')

# %%
# Function to calculate correlations and p-values
# def corr_cross_sites(df1, df2, corr_feature, sort_feature, n_boot=10000):
#     """
#     Calculate the bootstrap correlation and p-value between two dataframes.
#
#     Parameters:
#         df1 (pd.DataFrame): First dataframe.
#         df2 (pd.DataFrame): Second dataframe.
#         corr_feature (str): The column name for the feature to correlate.
#         sort_feature (str): The column name for the feature to sort by.
#         n_boot (int): Number of bootstrap iterations.
#
#     Returns:
#         (float, float): Mean correlation and mean p-value.
#     """
#     # Sort each dataframe by the sort feature
#     df1 = df1.sort_values(by=sort_feature).reset_index(drop=True)
#     df2 = df2.sort_values(by=sort_feature).reset_index(drop=True)
#
#     # Extract the features to correlate
#     Ser1 = df1[corr_feature]
#     Ser2 = df2[corr_feature]
#
#     # If sample sizes are equal, compute Pearson correlation directly
#     if len(Ser1) == len(Ser2):
#         corr, p_value = pearsonr(Ser1, Ser2)
#         return corr, p_value
#
#     # Identify the smaller and larger series
#     if len(Ser1) > len(Ser2):
#         larger, smaller = df1, df2
#     else:
#         larger, smaller = df2, df1
#
#     min_size = len(smaller)
#     corr_matrices = []
#     p_values = []
#
#     for i in range(n_boot):
#         # Sample the larger dataframe, then sort by the sort_feature
#         boot_larger = larger.sample(n=min_size, replace=True, random_state=i)
#         boot_larger_sorted = boot_larger.sort_values(by=sort_feature).reset_index(drop=True)
#
#         # Extract the sorted correlation feature for both dataframes
#         sorted_smaller = smaller[corr_feature].reset_index(drop=True)
#         sorted_boot_larger = boot_larger_sorted[corr_feature].reset_index(drop=True)
#
#         # Compute Pearson correlation and p-value
#         corr, p_value = pearsonr(sorted_smaller, sorted_boot_larger)
#         corr_matrices.append(corr)
#         p_values.append(p_value)
#
#     return np.mean(corr_matrices), np.mean(p_values)
#
# # List of all Stroop series
# Stroop_sites = {
#     'Greifswald': Stroop_SHIP,
#     'Liège': Stroop_Liege,
#     'Pittsburgh': Stroop_Pitts,
#     'Rotterdam': Stroop_EMC,
#     'San Diego': Stroop_VETSA
# }
#
# # Initialize empty matrices for correlations and p-values
# site_names = list(Stroop_sites.keys())
# n_sites = len(site_names)
#
# corr_matrix = np.zeros((n_sites, n_sites))
# p_matrix = np.zeros((n_sites, n_sites))
#
# # Pairwise correlation across all sites
# for i, site1 in enumerate(site_names):
#     for j, site2 in enumerate(site_names):
#         if i <= j:  # Calculate only for upper triangle and diagonal
#             corr, p_value = corr_cross_sites(Stroop_sites[site1], Stroop_sites[site2], 'Stroop_Test', 'Age_at_Scan')
#             corr_matrix[i, j] = corr
#             p_matrix[i, j] = p_value
#         else:
#             # Symmetric matrix: fill lower triangle
#             corr_matrix[i, j] = corr_matrix[j, i]
#             p_matrix[i, j] = p_matrix[j, i]
#
# # Convert to DataFrames for better readability
# corr_df = pd.DataFrame(corr_matrix, index=site_names, columns=site_names)
# p_df = pd.DataFrame(p_matrix, index=site_names, columns=site_names)
#
# # Display results
# print("Correlation Matrix Across Sites:")
# print(corr_df)
#
# print("\nP-Value Matrix Across Sites:")
# print(p_df)

# %%
# Flatten the upper triangle of the p-value matrix (excluding the diagonal)
# p_values_list = [
#     p_matrix[i, j]
#     for i in range(n_sites)
#     for j in range(i + 1, n_sites)  # Only upper triangle, exclude diagonal
# ]
#
# # Apply FDR correction (Benjamini-Hochberg)
# _, corrected_p_values, _, _ = multipletests(p_values_list, method='fdr_bh')
#
# # Reconstruct the corrected p-value matrix
# corrected_p_matrix = np.zeros_like(p_matrix)
#
# # Fill the corrected p-values into the matrix
# index = 0
# for i in range(n_sites):
#     for j in range(i + 1, n_sites):
#         corrected_p_matrix[i, j] = corrected_p_values[index]
#         corrected_p_matrix[j, i] = corrected_p_values[index]  # Symmetric matrix
#         index += 1
#
# # Convert to DataFrame for better readability
# corrected_p_df = pd.DataFrame(corrected_p_matrix, index=site_names, columns=site_names)
#
# # Display results
# print("Corrected P-Value Matrix (FDR):")
# print(corrected_p_df)

# %%
# Function to add significance labels
# def get_significance_label(p_value):
#     if p_value < 0.001:
#         return '***'
#     elif p_value < 0.01:
#         return '**'
#     elif p_value < 0.05:
#         return '*'
#     else:
#         return 'ns'
#
# # Create annotation matrix with significance labels for upper triangle
# annot_matrix = np.empty(corr_matrix.shape, dtype=object)
# annot_matrix[:] = ''  # Start with an empty matrix
#
# for i in range(len(site_names)):
#     for j in range(len(site_names)):
#         if i < j:  # Upper triangle only
#             p_val = p_matrix[i, j]
#             annot_matrix[i, j] = get_significance_label(p_val)
#         if i == j:
#             annot_matrix[i, j] = '-'
#
# # Set the lower triangle annotations to correlation values
# for i in range(len(site_names)):
#     for j in range(i):
#         annot_matrix[i, j] = f'{corr_matrix[i, j]:.2f}'
#
# # for corr_df, change the `Piitsburgh` to `Pittsburgh Executive Functioning`
# corr_df.rename(index={'Pittsburgh': 'Pittsburgh Executive'}, columns={'Pittsburgh': 'Pittsburgh Executive'}, inplace=True)
#
# # Plot heatmap
# plt.figure(figsize=(13, 10))
#
# sns.heatmap(
#     corr_df,
#     annot=annot_matrix,
#     cmap=plt.cm.RdBu_r,
#     fmt='',
#     vmin=-1, vmax=1,
#     linewidths=0.5,
#     square=True,
#     cbar_kws={'label': 'Correlation Coefficient'},
#     annot_kws={"size": 14}
# )
#
# plt.xticks(rotation=45, ha="right", fontsize=14)
# plt.yticks(rotation=0, fontsize=14)
# plt.xlabel('', fontsize=14)
# plt.ylabel('', fontsize=14)
# plt.title('Correlation Matrix of Stroop Test Score Across Sites', fontsize=20)
# plt.tight_layout()
# # save the plot as svg
# plt.savefig(result_save_path + 'Correlation_Matrix_Stroop.svg', format='svg')
# plt.show()
# plt.close()

# %%
"""
Correlation matrix of Memory test scores across sites
Matrix 1: Juelich session 1
Matrix 2: Juelich session 2
Matrix 3: Juelich session SD
"""
# Memory_SHIP = Memory_SHIP['Memory_Test'].sort_values()
# Memory_Liege = Memory_Liege['Memory_Test'].sort_values()
# Memory_KI = Memory_KI['Memory_Test'].sort_values()
# Memory_Pitts_1 = Memory_Pitts_1['Memory_Test'].sort_values()
# Memory_Pitts_2 = Memory_Pitts_2['Memory_Test'].sort_values()
# Memory_Juelich_Letter_HR_1 = Memory_Juelich_Letter_HR_1['Memory_Test'].sort_values()
# Memory_Juelich_Letter_Rtmean_1 = Memory_Juelich_Letter_Rtmean_1['Memory_Test'].sort_values()
# Memory_Juelich_Letter_Sensitivity_1 = Memory_Juelich_Letter_Sensitivity_1['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_HR_1 = Memory_Juelich_Spatial_HR_1['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_Rtmean_1 = Memory_Juelich_Spatial_Rtmean_1['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_Sensitivity_1 = Memory_Juelich_Spatial_Sensitivity_1['Memory_Test'].sort_values()
# Memory_Juelich_Letter_HR_2 = Memory_Juelich_Letter_HR_2['Memory_Test'].sort_values()
# Memory_Juelich_Letter_Rtmean_2 = Memory_Juelich_Letter_Rtmean_2['Memory_Test'].sort_values()
# Memory_Juelich_Letter_Sensitivity_2 = Memory_Juelich_Letter_Sensitivity_2['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_HR_2 = Memory_Juelich_Spatial_HR_2['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_Rtmean_2 = Memory_Juelich_Spatial_Rtmean_2['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_Sensitivity_2 = Memory_Juelich_Spatial_Sensitivity_2['Memory_Test'].sort_values()
# Memory_Juelich_Letter_HR_SD = Memory_Juelich_Letter_HR_SD['Memory_Test'].sort_values()
# Memory_Juelich_Letter_Rtmean_SD = Memory_Juelich_Letter_Rtmean_SD['Memory_Test'].sort_values()
# Memory_Juelich_Letter_Sensitivity_SD = Memory_Juelich_Letter_Sensitivity_SD['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_HR_SD = Memory_Juelich_Spatial_HR_SD['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_Rtmean_SD = Memory_Juelich_Spatial_Rtmean_SD['Memory_Test'].sort_values()
# Memory_Juelich_Spatial_Sensitivity_SD = Memory_Juelich_Spatial_Sensitivity_SD['Memory_Test'].sort_values()
# Memory_VETSA_Digit = Memory_VETSA_Digit['Memory_Test'].sort_values()
# Memory_VETSA_Letter = Memory_VETSA_Letter['Memory_Test'].sort_values()

# %%
# Memory_sites_1 = {
#     'Greifswald': Memory_SHIP,
#     'Liège': Memory_Liege,
#     'Stockholm': Memory_KI,
#     'Pittsburgh Letter': Memory_Pitts_1,
#     'Piitsburgh Spatial': Memory_Pitts_2,
#     'Jülich Letter HR': Memory_Juelich_Letter_HR_1,
#     'Jülich Letter Rtmean': Memory_Juelich_Letter_Rtmean_1,
#     'Jülich Letter Sensitivity': Memory_Juelich_Letter_Sensitivity_1,
#     'Jülich Spatial HR': Memory_Juelich_Spatial_HR_1,
#     'Jülich Spatial Rtmean': Memory_Juelich_Spatial_Rtmean_1,
#     'Jülich Spatial Sensitivity': Memory_Juelich_Spatial_Sensitivity_1,
#     'San Diego Letter': Memory_VETSA_Letter,
#     'San Diego Digit': Memory_VETSA_Digit
# }
#
# # Initialize empty matrices for correlations and p-values
# site_names_1 = list(Memory_sites_1.keys())
# n_sites_1 = len(site_names_1)
#
# corr_matrix_1 = np.zeros((n_sites_1, n_sites_1))
# p_matrix_1 = np.zeros((n_sites_1, n_sites_1))
#
# # Pairwise correlation across all sites
# for i, site1 in enumerate(site_names_1):
#     for j, site2 in enumerate(site_names_1):
#         if i <= j:  # Calculate only for upper triangle and diagonal
#             corr, p_value = corr_cross_sites(Memory_sites_1[site1], Memory_sites_1[site2], 'Memory_Test', 'Age_at_Scan')
#             corr_matrix_1[i, j] = corr
#             p_matrix_1[i, j] = p_value
#         else:
#             # Symmetric matrix: fill lower triangle
#             corr_matrix_1[i, j] = corr_matrix_1[j, i]
#             p_matrix_1[i, j] = p_matrix_1[j, i]
#
# # Convert to DataFrames for better readability
# corr_df_1 = pd.DataFrame(corr_matrix_1, index=site_names_1, columns=site_names_1)
# p_df_1 = pd.DataFrame(p_matrix_1, index=site_names_1, columns=site_names_1)
#
# # Display results
# print("Correlation Matrix Across Sites:")
# print(corr_df_1)
#
# print("\nP-Value Matrix Across Sites:")
# print(p_df_1)
#
# # %%
# # Create annotation matrix with significance labels for upper triangle
# annot_matrix = np.empty(corr_matrix_1.shape, dtype=object)
# annot_matrix[:] = ''  # Start with an empty matrix
#
# for i in range(len(site_names_1)):
#     for j in range(len(site_names_1)):
#         if i < j:  # Upper triangle only
#             p_val = p_matrix_1[i, j]
#             annot_matrix[i, j] = get_significance_label(p_val)
#         if i == j:
#             annot_matrix[i, j] = '-'
#
# # Set the lower triangle annotations to correlation values
# for i in range(len(site_names_1)):
#     for j in range(i):
#         annot_matrix[i, j] = f'{corr_matrix_1[i, j]:.2f}'
#
# # Plot heatmap
# plt.figure(figsize=(12, 10))
#
# sns.heatmap(
#     corr_df_1,
#     annot=annot_matrix,
#     cmap=plt.cm.RdBu_r,
#     fmt='',
#     vmin=-1, vmax=1,
#     linewidths=0.5,
#     square=True,
#     cbar_kws={'label': 'Correlation Coefficient'},
#     annot_kws={"size": 14}
# )
#
# plt.xticks(rotation=45, ha="right", fontsize=14)
# plt.yticks(rotation=0, fontsize=14)
# plt.xlabel('', fontsize=14)
# plt.ylabel('', fontsize=14)
# plt.title('Correlation Matrix of Memory Test Score Across Sites', fontsize=20)
# plt.tight_layout()
# # save the plot as svg
# plt.savefig(result_save_path + 'Correlation_Matrix_Memory.svg', format='svg')
# plt.show()
# plt.close()
