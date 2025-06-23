import re
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')

def parse_log_file(log_path):
    """
    Parses the training log file to extract step and loss values.
    """
    log_pattern = re.compile(r"Step \[(\d+)/\d+\], Loss: ([\d\.]+)")
    steps = []
    losses = []

    print(f"--> Reading log file from: {log_path}")
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    steps.append(step)
                    losses.append(loss)
    except FileNotFoundError:
        print(f"!! ERROR: Log file not found at '{log_path}'. Make sure the path is correct.")
        return None, None

    print(f"--> Found {len(steps)} data points.")
    return steps, losses

def plot_loss_graph(steps, losses, y_scale='linear', output_file='training_loss.png'):
    """
    Creates and saves a plot of loss vs. steps, with an option for a log scale y-axis.
    """
    if not steps or not losses:
        print("!! No data to plot.")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(steps, losses, label='Training Loss', color='deepskyblue', alpha=0.7, linestyle='-')
    
    # --- Optional: Add a smoothed line to see the trend better ---
    if len(steps) > 50:
        df = pd.DataFrame({'loss': losses})
        smoothed_loss = df['loss'].rolling(window=50, min_periods=1).mean()
        plt.plot(steps, smoothed_loss, label='Smoothed Loss (50-step avg)', color='orangered', linewidth=2)

    plt.title('Training Loss vs. Steps', fontsize=16)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(f'Loss ({y_scale.capitalize()} Scale)', fontsize=12) # Dynamic Y-label
    plt.legend()
    
    # =======================================================
    # ===> THE KEY CHANGE TO SET THE SCALE <===
    # =======================================================
    if y_scale == 'log':
        #plt.yscale('log')
        plt.xscale('log')
        # When using a log scale, it's often better to not have grid lines on the minor ticks
        plt.grid(True, which='both', ls='-')
    else:
        plt.grid(True)
    
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"--> Graph saved successfully to: {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss from a log file.")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    default_log_path = os.path.join(project_root, '', 'training.log')
    
    parser.add_argument('--log_file', type=str, default=default_log_path,
                        help='Path to the training log file.')
    parser.add_argument('--output', type=str, default='training_loss.png',
                        help='Path to save the output plot image.')
    # ===> ADDED NEW ARGUMENT FOR SCALE <===
    parser.add_argument('--yscale', type=str, default='linear', choices=['linear', 'log'],
                        help="Set the y-axis scale ('linear' or 'log'). Default is linear.")
    parser.add_argument('--xscale', type=str, default='linear', choices=['linear', 'log'],
                        help="Set the y-axis scale ('linear' or 'log'). Default is linear.")
    args = parser.parse_args()
    
    steps, losses = parse_log_file(args.log_file)
    plot_loss_graph(steps, losses, args.yscale, args.output)