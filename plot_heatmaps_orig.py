import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the accuracy results from the JSON file
with open('all_accuracies385.json', 'r') as f:
    all_results = json.load(f)

# Set the seaborn style and font scale
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Create the heatmaps directory if it doesn't exist
if not os.path.exists('heatmaps385'):
    os.makedirs('heatmaps385')

def custom_annot(val):
    if val == 100:
        return '100'
    else:
        return f'{val:.1f}'
# Iterate over each folder's results and generate a heatmap
for folder_name, accs in all_results.items():
    # Convert the accuracy dictionary to a pandas DataFrame
    df = pd.DataFrame(accs).transpose() * 100  # Convert to percentages

    # Generate the heatmap
    #ax = sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True, linewidths=0.5, annot_kws={"size": 8})
    #ax = sns.heatmap(df, annot=True, fmt=".1f", cbar=True, cmap="crest", linewidths=0.5, annot_kws={"size": 8})
    #ax = sns.heatmap(df, annot=True, fmt=".1f", cbar=True, cmap="crest", linewidths=0.5, annot_kws={"size": 5})
    cmap = 'crest'
    cmap = 'Greens'
    cmap = 'Blues'
    cmap = 'viridis'
    cmap = 'cividis'
    #for cmap in ['crest', 'Greens', 'Blues', 'viridis', 'cividis', 'Purples', 'Reds', 'plasma', 'magma', 'inferno', 'RdYlGn', 'YlGn']:
    for cmap in ['RdYlGn']:
    #for cmap in ['YlGn']:
        plt.figure(figsize=(12, 7.4))
        ax = sns.heatmap(df, annot=df.applymap(custom_annot), fmt="", cbar=True, cmap=cmap, linewidths=0.5, annot_kws={"size": 11}, cbar_kws={"shrink": 0.75}, vmin=0)
        #ax.set_aspect('equal', 'box')
        ax.set_xlabel('Digits in Number 1')
        ax.set_ylabel('Digits in Number 2')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', which='both', length=0)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Accuracy (%)')
        plt.title(f'Accuracy of Implicit CoT (Stepwise Internalization)', fontsize=18)
        
        # Ensure the heatmap is square

        # Save the heatmap to a file
        #plt.tight_layout()
        plt.savefig(f'heatmap_orig.png')
        plt.close()

print("Heatmaps generated and saved in the 'heatmaps2' folder.")

