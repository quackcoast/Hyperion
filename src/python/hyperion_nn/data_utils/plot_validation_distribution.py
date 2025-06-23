# python -m hyperion_nn.data_utils.plot_validation_distribution
# python -m hyperion_nn.data_utils.plot_validation_distribution --file data/validation/validation_results_step_30000.pt

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import hyperion_nn.config as config

def plot_distribution(file_path):
    """
    Loads validation results and plots the distribution of outcomes
    for predicted values.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # --- Load Data ---
    data = torch.load(file_path)
    predictions = data['predictions'].squeeze().numpy()
    targets = data['targets'].squeeze().numpy()

    # --- Bin the Data ---
    # Create 20 bins from -1.0 to 1.0
    num_bins = 20
    bins = np.linspace(-1, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Dictionaries to hold the counts for each bin
    win_counts = {i: 0 for i in range(num_bins)}
    loss_counts = {i: 0 for i in range(num_bins)}
    draw_counts = {i: 0 for i in range(num_bins)}

    # Digitize the predictions to find which bin each falls into
    binned_predictions = np.digitize(predictions, bins) - 1

    # --- Count Outcomes per Bin ---
    for i in range(len(binned_predictions)):
        bin_index = binned_predictions[i]
        
        # Ensure bin_index is valid (it can be out of bounds if pred is exactly 1.0)
        if 0 <= bin_index < num_bins:
            target = targets[i]
            if target == 1:
                win_counts[bin_index] += 1
            elif target == -1:
                loss_counts[bin_index] += 1
            elif target == 0:
                draw_counts[bin_index] += 1

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Convert counts to lists for plotting
    win_line = [win_counts[i] for i in range(num_bins)]
    loss_line = [loss_counts[i] for i in range(num_bins)]
    draw_line = [draw_counts[i] for i in range(num_bins)]

    ax.plot(bin_centers, win_line, marker='o', linestyle='-', color='green', label='Actual Wins')
    ax.plot(bin_centers, loss_line, marker='o', linestyle='-', color='red', label='Actual Losses')
    ax.plot(bin_centers, draw_line, marker='o', linestyle='-', color='gray', label='Actual Draws')

    # --- Formatting ---
    step_number = os.path.basename(file_path).split('_')[-1].split('.')[0]
    ax.set_title(f'Value Head Calibration at Training Step {step_number}', fontsize=16)
    ax.set_xlabel('Model Predicted Value', fontsize=12)
    ax.set_ylabel('Number of Games', fontsize=12)
    ax.legend()
    ax.set_xticks(np.linspace(-1, 1, 11)) # Set x-axis ticks for clarity

    # --- Save the Plot ---
    os.makedirs(config.PathsConfig.VALIDATION_DIR, exist_ok=True)
    plot_filename = os.path.join(config.PathsConfig.VALIDATION_DIR, f'value_distribution_step_{step_number}1.png')
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot value head calibration from a validation results file.")
    # Argument to specify a single file
    parser.add_argument('--file', type=str, help="Path to a specific validation_results_step_*.pt file.")
    args = parser.parse_args()

    if args.file:
        plot_distribution(args.file)
    else:
        # If no file is specified, find the latest one
        validation_dir = config.PathsConfig.VALIDATION_DIR
        list_of_files = glob.glob(os.path.join(validation_dir, 'validation_results_step_*.pt'))
        if not list_of_files:
            print("No validation result files found. Run training with validation first.")
        else:
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"No file specified. Plotting latest results from: {latest_file}")
            plot_distribution(latest_file)