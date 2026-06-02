import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ridgeplot import ridgeplot

# %%
import matplotlib.pyplot as plt

# Define the colors
colors = ['#88A7B5', '#AFCECD', '#D4E3DE', '#D7E7E8', '#E5E8E9', '#F1C3C7', '#E2D4E8']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 2))

# Plot the colors as rectangles
for i, color in enumerate(colors):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.text(i + 0.5, -0.5, color, ha='center', va='center', fontsize=10, color='black')

# Remove the axis
ax.set_xlim(0, len(colors))
ax.set_ylim(0, 1)
ax.axis('off')

# Display the palette
plt.show()

# %%
# Original colors provided by the user
original_palette = ['#88A7B5', '#AFCECD', '#D4E3DE', '#D7E7E8', '#E5E8E9', '#F1C3C7', '#E2D4E8']

# Option 1: All colors revised
revised_all_colors = ['#6F97A5', '#ABD7E0', '#93C4C0', '#B9D3CE', '#DEE3E6', '#F3B6BC', '#D7C5E5']

# Option 2: Middle three revised
revised_middle_colors = ['#88A7B5', '#92BFBF', '#C7DDD8', '#A2D6E2', '#E5E8E9', '#F1C3C7', '#E2D4E8']

# Visualization function
def visualize_palette(colors, title, ax):
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(i + 0.5, -0.5, color, ha='center', va='center', fontsize=10, color='black')
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=12)

# Plot original, revised all, and revised middle palettes
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

visualize_palette(original_palette, "Original Colors", axs[0])
visualize_palette(revised_all_colors, "Revised All Colors", axs[1])
visualize_palette(revised_middle_colors, "Revised Middle Three Colors", axs[2])

plt.tight_layout()
plt.show()

# %%
"""
Color maps of all sites
SHIP-Trend: SLATE #6F97A5
Liege: BLUE-GREEN #93C4C0
KI: PASTEL GREEN #B9D3CE
Juelich: #ABD7E0
Pitts: STONE #DEE3E6
EMC: PASTEL PINK #F3B6BC
VETSA: LAVENDER #D7C5E5
"""
# Load the data
data_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
df_SHIP_renamed = pd.read_csv(data_path + 'SHIP_Trend_dataset_renamed.csv')
df_Liege_renamed = pd.read_csv(data_path + 'Liege_dataset_renamed.csv')
df_KI_renamed = pd.read_csv(data_path + 'KI_dataset_renamed.csv')
df_Pitts_renamed = pd.read_csv(data_path + 'Pitts_dataset_renamed_target_transformed.csv')

# %%
def ridge_plot(df, x_variable, y_variable, x_label=None, color_palette=None):
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
    plt.figure(figsize=(10, 8))

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

    # save figure as svg
    plt.savefig(save_path + f'{x_variable}_ridge_plot.svg', format='svg')
    # Show the final ridge plot
    plt.show()
    plt.close()


# %%
"""
make ridge plot of 'Age_at_Scan' of all datasets
"""
# make dataframe
df_SHIP_age = df_SHIP_renamed[['Age_at_Scan']]
df_SHIP_age['Dataset'] = 'Greifswald'
df_Liege_age = df_Liege_renamed[['Age_at_Scan']]
df_Liege_age['Dataset'] = 'Liège'
df_KI_age = df_KI_renamed[['Age_at_Scan']]
df_KI_age['Dataset'] = 'Stockholm'
df_Pitts_age = df_Pitts_renamed[['Age_at_Scan']]
df_Pitts_age['Dataset'] = 'Pittsburgh'

df_age = pd.concat([df_SHIP_age, df_Liege_age, df_KI_age, df_Pitts_age])

custom_colors = ['#88A7B5', '#AFCECD', '#D4E3DE', '#E5E8E9']
ridge_plot(df_age, "Age_at_Scan", "Dataset", x_label="Age", color_palette=custom_colors)

# %%
"""
make ridge plot of 'Stroop_Test' of df_SHIP_renamed, df_Liege_renamed
"""
# make dataframe
df_SHIP_stroop = df_SHIP_renamed[['Stroop_Test']]
df_SHIP_stroop['Dataset'] = 'Greifswald'
df_Liege_stroop = df_Liege_renamed[['Stroop_Test']]
df_Liege_stroop['Dataset'] = 'Liège'

df_stroop = pd.concat([df_SHIP_stroop, df_Liege_stroop])

custom_colors = ['#88A7B5', '#AFCECD']
ridge_plot(df_stroop, 'Stroop_Test', 'Dataset', x_label='Stroop interference score (reaction time)', color_palette=custom_colors)

# %%
"""
make ridge plot of 'Memory_Test' of df_SHIP_renamed, df_Liege_renamed, df_KI_renamed, df_Pitts_renamed
"""
# make dataframe
df_SHIP_renamed['Memory_Test'] = df_SHIP_renamed['Memory_Test'].apply(lambda x: x / 16) * 100
df_SHIP_memory = df_SHIP_renamed[['Memory_Test']]
df_SHIP_memory['Dataset'] = 'Greifswald'
df_Liege_renamed['Memory_Test'] = df_Liege_renamed['Memory_Test'] * 100
df_Liege_memory = df_Liege_renamed[['Memory_Test']]
df_Liege_memory['Dataset'] = 'Liège'
df_KI_memory = df_KI_renamed[['Memory_Test']]
df_KI_memory['Dataset'] = 'Stockholm'
# df_Pitts_renamed rename 'Memory_Test2' to 'Memory_Test'
df_Pitts_renamed.rename(columns={'Memory_Test2': 'Memory_Test'}, inplace=True)
df_Pitts_memory = df_Pitts_renamed[['Memory_Test']]
df_Pitts_memory['Dataset'] = 'Pittsburgh'

df_memory = pd.concat([df_SHIP_memory, df_Liege_memory, df_KI_memory, df_Pitts_memory])

custom_colors = ['#88A7B5', '#AFCECD', '#D4E3DE', '#E5E8E9']
ridge_plot(df_memory, 'Memory_Test', 'Dataset', x_label='Memory score (%)', color_palette=custom_colors)

# %%
# Set the seaborn theme for better visualization
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# figsize
plt.figure(figsize=(10, 8))
# Initialize the FacetGrid object with Dataset as the row variable
pal = sns.cubehelix_palette(len(df_age['Dataset'].unique()), rot=-.25, light=.7)
g = sns.FacetGrid(df_age, row="Dataset", hue="Dataset", aspect=6, height=2, palette=pal)

# Plot the KDEs (Kernel Density Estimates)
g.map(sns.kdeplot, "Age_at_Scan",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)

# Overlay white lines for contrast
g.map(sns.kdeplot, "Age_at_Scan", clip_on=False, color="w", lw=2, bw_adjust=.5)

# Add a reference line at y=0
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define a function to add labels inside each facet
def label(x, color, label):
    ax = plt.gca()
    ax.text(-0.1, .2, label, fontweight="bold", color=color, fontsize=18,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "Age_at_Scan")

# Adjust the spacing between plots
g.figure.subplots_adjust(hspace=-.3)

# Remove titles and axis labels that may overlap
g.set_titles("")
g.set(yticks=[], ylabel="")

# Change x-axis label size and style
g.set(xlabel="Age")

# Adjust tick and label size using Matplotlib
for ax in g.axes.flat:
    ax.tick_params(axis='x', labelsize=18)  # Change x-axis tick label size
    ax.set_xlabel("Age", fontsize=20)  # Set x-axis label size and style

g.despine(bottom=True, left=True)

# Show the final ridge plot
plt.show()
plt.close()
