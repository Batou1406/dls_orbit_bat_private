import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os

import matplotlib
print(matplotlib.get_backend())


# File paths
file_paths = {
    'cost': 'live_variable/cost.csv',
}

winning_policy_path = 'live_variable/winning_policy.csv'

# Initialize the plot
fig, ax = plt.subplots()

# Initialize lines for each file
# lines = {}
# for label in file_paths.keys():
#     line, = ax.plot([], [], lw=2, label=label)
#     lines[label] = line

line_segments = LineCollection([], linewidths=2)
ax.add_collection(line_segments)

# Set up the plot labels and limits
ax.set_xlabel('Iteration')
ax.set_ylabel('Best trajectory cost')
ax.legend(loc='best')

ax.set_xlim(-0.1, 100.1)
ax.set_ylim(0.25, 0.50)

# def init():
#     for line in lines.values():
#         line.set_data([], [])
#     return lines.values()

def update(frame):
    try:
        for label, file_path in file_paths.items():
            if os.path.exists(file_path):
                # Load the data from CSV file
                data = np.loadtxt(file_path, delimiter=',')
                # Select only the third line (row index 2)
                data_to_plot = data
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                # lines[label].set_data(x, y)

            else:
                print(f"File {file_path} does not exist.")

        c = np.loadtxt(winning_policy_path, delimiter=',')

        # Split the line into segments
        segments = [[(x[i], y[i]), (x[i+1], y[i+1])] for i in range(len(x)-1)]
        
        # Create an array of colors based on the `c` variable
        colors = plt.cm.jet(c[:-1] / float(max(c)))
        
        # Update the LineCollection
        line_segments.set_segments(segments)
        line_segments.set_color(colors)
        
        y_min = min(0.0,np.min(y) - 1.0)
        y_max = max(200,np.max(y) + 1.0 )
        ax.set_xlim(0, len(data_to_plot) + 5)
        ax.set_ylim(y_min, y_max)

        # Adjust x and y limits based on the data
        # all_y_data = [lines[label].get_ydata() for label in file_paths.keys()]
        # if all_y_data:
        #     y_min = min(0.0,min(np.min(y) - 1.0 for y in all_y_data))
        #     y_max = max(200,max(np.max(y) + 1.0 for y in all_y_data))
        #     ax.set_xlim(0, len(data_to_plot) + 5)
        #     ax.set_ylim(y_min, y_max)

    except Exception as e:
        print(f"Error reading or processing the file: {e}")

    # return lines.values()
    return line_segments,

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    # init_func=init,
    blit=True,
    interval=100,
    cache_frame_data=False,
)

plt.show()

