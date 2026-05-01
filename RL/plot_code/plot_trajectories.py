import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_three_condition_trajectories(file_paths, output_filename="attack_freq_comparison.png"):
    """
    Draws a 3x3 trajectory plot based on three CSV files.
    Rows: Different conditions (files)
    Columns: Position, Speed, Acceleration
    """
    
    # Sort file paths to ensure order 0.1, 0.3, 0.5 if possible, or just use provided order
    # The user provided: 0.5, 0.1, 0.3. 
    # Let's try to parse frequency from filename to sort them, or just respect the list if we can't.
    # Filenames: ddpg_attack_freq_0.5_..., ddpg_attack_freq_0.1_..., ddpg_attack_freq_0.3_...
    
    # Helper to extract frequency for sorting
    def get_freq(path):
        try:
            # simple extraction assuming format ...freq_X.X...
            parts = path.split('_')
            for i, p in enumerate(parts):
                if p == 'freq':
                    return float(parts[i+1])
        except:
            return 0
        return 0

    # Sort files by frequency
    sorted_paths = sorted(file_paths, key=get_freq)
    
    # Define plot style
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    # Create 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.05,
                        hspace=0.30, wspace=0.15)

    # Colors
    hv_color = '#d62728'  # Red for HV
    cav_color = '#1f77b4' # Blue for CAV
    
    # Process each file
    for idx, file_path in enumerate(sorted_paths):
        print(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Extract frequency for title
        freq = get_freq(file_path)
        condition_title = f"Attack Frequency {freq}"

        # Get unique vehicle IDs
        vehicle_ids = df['VehicleID'].unique()
        
        # Get axes for this row
        ax_pos = axes[idx, 0]
        ax_vel = axes[idx, 1]
        ax_acc = axes[idx, 2]

        # Plot each vehicle
        for vid in vehicle_ids:
            vehicle_data = df[df['VehicleID'] == vid]
            
            # Determine type and style
            v_type = vehicle_data['VehicleType'].iloc[0]
            is_cav = (v_type == 'CAV')
            
            color = cav_color if is_cav else hv_color
            line_style = '--' if is_cav else '-'
            linewidth = 1.3
            
            # Data
            time_step = vehicle_data['Step'] # Or Time(s) / dt. Using Step from CSV.
            # If Step is not available, use Time(s) * 10 (assuming dt=0.1)
            # The CSV header showed "Step", so we use that.
            
            pos = vehicle_data['Position(m)']
            vel = vehicle_data['Speed(m/s)']
            acc = vehicle_data['Acceleration(m/s2)']
            
            # Plot
            ax_pos.plot(time_step, pos, color=color, linestyle=line_style, linewidth=linewidth)
            ax_vel.plot(time_step, vel, color=color, linestyle=line_style, linewidth=linewidth)
            ax_acc.plot(time_step, acc, color=color, linestyle=line_style, linewidth=linewidth)

        # Set Row Title (Condition) - using text on the left or title on the center plot?
        # Reference used (a), (b) at bottom. We can put title on top of the center plot of the row, 
        # or just set title for each subplot in the first row?
        # Actually, usually "Row" represents condition. Let's add a text label or title.
        # Let's add a title to the middle column of each row to indicate the condition

        # Labels
        ax_pos.set_ylabel("Position (m)")
        ax_vel.set_ylabel("Speed (m/s)")
        ax_acc.set_ylabel("Acceleration (m/s$^2$)")
        
        # X labels only for bottom row? Or all?
        # Reference set for all.
        ax_pos.set_xlabel("Time Step")
        ax_vel.set_xlabel("Time Step")
        ax_acc.set_xlabel("Time Step")

        # Grids
        for ax in (ax_pos, ax_vel, ax_acc):
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            
            # Set limits if needed to match reference style
            # Reference: vel ylim dynamic, acc ylim -3 to 3
            ax_acc.set_ylim(-4, 4) # Slightly wider to catch spikes
            
        # Legend only on first row, first plot (or where appropriate)
        if idx == 0:
            legend_elements = [
                plt.Line2D([0], [0], color=cav_color, linestyle='--', linewidth=1.5),
                plt.Line2D([0], [0], color=hv_color, linestyle='-', linewidth=1.5)
            ]
            legend_labels = ['CAV', 'HV']
            ax_pos.legend(legend_elements, legend_labels, loc='upper right')

    # Add row labels (a), (b), (c)
    row_labels = ['(a)', '(b)', '(c)']
    for i, label in enumerate(row_labels):
        # Position text at the bottom center of the row
        # We can use the middle axis of the row
        row_axes = axes[i]
        # Get the bounding box of the middle axis
        bbox = row_axes[1].get_position()
        # Calculate center x, and a y slightly below the axis
        # But we have x-labels. 
        # Reference code: fig.text(0.5, row_bottom - 0.08, ...)
        
        # Let's just put it below the middle plot's x-label
        # Or follow reference exactly:
        row_bottom = min(ax.get_position().y0 for ax in row_axes)
        fig.text(0.52, row_bottom - 0.05, label,
                 fontsize=16, fontweight='bold', ha='center', va='top')

    # Save
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_filename, dpi=350, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    # plt.show() # Commented out for batch running

if __name__ == "__main__":
    # Define files
    base_dir = r"c:\Users\12112\Desktop\ddd\examples\outputs"
    files = [
        os.path.join(base_dir, "ddpg_attack_freq_0.3_all_vehicles_trajectory.csv"),
        os.path.join(base_dir, "Feedback_Control_attack_freq_0.3_all_vehicles_trajectory.csv"),
        os.path.join(base_dir, "ddpg_attack_freq_0.3_all_vehicles_trajectory_shi.csv")
    ]
    
    output_path = os.path.join(base_dir, "ddpg_attack_freq_comparison_3x3.png")
    
    plot_three_condition_trajectories(files, output_path)


