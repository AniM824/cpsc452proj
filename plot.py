import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir = 'results'

datasets = ['cifar10', 'cifar100', 'emnist', 'isic', 'tem']

plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

for i, dataset in enumerate(datasets):
    dataset_dir = os.path.join(results_dir, dataset)
    
    if not os.path.exists(dataset_dir):
        print(f"Directory for {dataset} not found. Skipping...")
        continue
    
    baseline_file = None
    equivariant_file = None
    
    for file in os.listdir(dataset_dir):
        if file.endswith('.csv'):
            if 'baseline' in file:
                baseline_file = os.path.join(dataset_dir, file)
            elif 'equivariant' in file:
                equivariant_file = os.path.join(dataset_dir, file)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{dataset.upper()} Training Metrics Comparison', fontsize=16)
    
    ax_loss = axs[0]
    ax_loss.set_title('Training Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    
    ax_acc = axs[1]
    ax_acc.set_title('Training Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    
    if baseline_file:
        try:
            baseline_data = pd.read_csv(baseline_file)
            ax_loss.plot(baseline_data['Epoch'], baseline_data['Loss'], 'b-', label='Baseline')
            ax_acc.plot(baseline_data['Epoch'], baseline_data['Accuracy'], 'b-', label='Baseline')
        except Exception as e:
            print(f"Error reading baseline file {baseline_file}: {e}")
    
    if equivariant_file:
        try:
            equivariant_data = pd.read_csv(equivariant_file)
            ax_loss.plot(equivariant_data['Epoch'], equivariant_data['Loss'], 'r-', label='Equivariant')
            ax_acc.plot(equivariant_data['Epoch'], equivariant_data['Accuracy'], 'r-', label='Equivariant')
        except Exception as e:
            print(f"Error reading equivariant file {equivariant_file}: {e}")
    
    if baseline_file or equivariant_file:
        ax_loss.legend()
        ax_acc.legend()
    else:
        fig.text(0.5, 0.5, 'No data found for baseline or equivariant models.', ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(plots_dir, f'{dataset}_metrics_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close(fig)
