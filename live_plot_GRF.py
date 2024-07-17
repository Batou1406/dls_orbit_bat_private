import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# File paths
file_paths = {
    'FL': 'F_best_FL.csv',
    'FR': 'F_best_FR.csv',
    'RL': 'F_best_RL.csv',
    'RR': 'F_best_RR.csv'
}

# Initialize the plot
fig, ax = plt.subplots()

# Initialize lines for each file
lines = {}
for label in file_paths.keys():
    line, = ax.plot([], [], lw=2, label=label)
    lines[label] = line

# Set up the plot labels and limits
ax.set_xlabel('Iteration')
ax.set_ylabel('Force Z [N]')
ax.legend(loc='best')

ax.set_xlim(-0.1, 3.1)
ax.set_ylim(-70, 70)

def init():
    for line in lines.values():
        line.set_data([], [])
    return lines.values()

def update(frame):
    try:
        for label, file_path in file_paths.items():
            if os.path.exists(file_path):
                # Load the data from CSV file
                data = np.loadtxt(file_path, delimiter=',')
                # Select only the third line (row index 2)
                data_to_plot = data[2, :]
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                lines[label].set_data(x, y)
                # lines[label].set_ydata(y)
            else:
                print(f"File {file_path} does not exist.")
        
        # Adjust x and y limits based on the data
        # all_y_data = [lines[label].get_ydata() for label in file_paths.keys()]
        # if all_y_data:
        #     y_min = min(np.min(y) for y in all_y_data)
        #     y_max = max(np.max(y) for y in all_y_data)
        #     ax.set_xlim(0, len(data_to_plot) - 1)
        #     ax.set_ylim(y_min, y_max)

    except Exception as e:
        print(f"Error reading or processing the file: {e}")

    return lines.values()

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    blit=True,
    interval=100,
    cache_frame_data=False,
)

plt.show()

