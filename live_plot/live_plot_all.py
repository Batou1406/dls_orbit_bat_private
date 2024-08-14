import matplotlib
# matplotlib.use('GTK4Agg') 
matplotlib.use('qtagg') 
print(matplotlib.get_backend())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os


# File paths
file_paths_GRF = {
    'FL': 'live_variable/F_best_FL.csv',
    'FR': 'live_variable/F_best_FR.csv',
    'RL': 'live_variable/F_best_RL.csv',
    'RR': 'live_variable/F_best_RR.csv'
}

file_paths_cost = {
    'cost': 'live_variable/cost.csv',
}
winning_policy_path = 'live_variable/winning_policy.csv'

file_paths_height = {
    'height'    : 'live_variable/height.csv',
    'height_ref': 'live_variable/height_ref.csv'
}

file_paths_foot_x = {
    'FL_foot x': 'live_variable/FL_foot.csv',
    'FR_foot x': 'live_variable/FR_foot.csv',
    'RL_foot x': 'live_variable/RL_foot.csv',
    'RR_foot x': 'live_variable/RR_foot.csv',
    # 'FL_foot x ref': 'live_variable/FL_foot_ref.csv',
    # 'FR_foot x ref': 'live_variable/FR_foot_ref.csv',
    # 'RL_foot x ref': 'live_variable/RL_foot_ref.csv',
    # 'RR_foot x ref': 'live_variable/RR_foot_ref.csv'
}

file_paths_foot_y = {
    'FL_foot y': 'live_variable/FL_foot.csv',
    'FR_foot y': 'live_variable/FR_foot.csv',
    'RL_foot y': 'live_variable/RL_foot.csv',
    'RR_foot y': 'live_variable/RR_foot.csv',
    # 'FL_foot y ref': 'live_variable/FL_foot_ref.csv',
    # 'FR_foot y ref': 'live_variable/FR_foot_ref.csv',
    # 'RL_foot y ref': 'live_variable/RL_foot_ref.csv',
    # 'RR_foot y ref': 'live_variable/RR_foot_ref.csv'
}

file_paths_foot_z = {
    'FL_foot z': 'live_variable/FL_foot.csv',
    'FR_foot z': 'live_variable/FR_foot.csv',
    'RL_foot z': 'live_variable/RL_foot.csv',
    'RR_foot z': 'live_variable/RR_foot.csv',
    # 'FL_foot z ref': 'live_variable/FL_foot_ref.csv',
    # 'FR_foot z ref': 'live_variable/FR_foot_ref.csv',
    # 'RL_foot z ref': 'live_variable/RL_foot_ref.csv',
    # 'RR_foot z ref': 'live_variable/RR_foot_ref.csv'
}


# Initialize the plot
fig, axs = plt.subplots(2,3)

axGRF = axs[0,0]
axHeight = axs[1,0]
axCost = axs[0,1]
axFootZ = axs[1,1]
axFootX = axs[0,2]
axFootY = axs[1,2]

# ------------- GRF --------------
# Initialize lines for each file
lines_grf = {}
for label in file_paths_GRF.keys():
    line, = axGRF.plot([], [], lw=2, label=label)
    lines_grf[label] = line

# Set up the plot labels and limits
axGRF.set_xlabel('Iteration')
axGRF.set_ylabel('Force Z [N]')
axGRF.legend(loc='best')
axGRF.set_title('Ground Reaction Forces')

axGRF.set_xlim(-1.2, 2.2)
axGRF.set_ylim(-70, 70)

# ---------------- Cost --------------
# lines_cost = {}
# for label in file_paths_cost.keys():
#     line, = axCost.plot([], [], lw=2, label=label)
#     lines_cost[label] = line

lines_cost = LineCollection([], linewidths=2)
axCost.add_collection(lines_cost)

# Set up the plot labels and limits
axCost.set_xlabel('Iteration')
axCost.set_ylabel('Best trajectory cost')
axCost.legend(loc='best')
axCost.set_title('Best Trajectory Cost')

axCost.set_xlim(-0.1, 100.1)
axCost.set_ylim(0.0, 100.0)

# -------------- Height -----------
lines_height = {}
for label in file_paths_height.keys():
    line, = axHeight.plot([], [], lw=2, label=label)
    lines_height[label] = line

# Set up the plot labels and limits
axHeight.set_xlabel('Iteration')
axHeight.set_ylabel('Robot\'s Height [m]')
axHeight.legend(loc='best')
axHeight.set_title('Robot\'s Height')

axHeight.set_xlim(-0.1, 100.1)
axHeight.set_ylim(0.30, 0.40)

# ------------- Foot X --------------
lines_foot_x = {}
colors_foot_x = {}

# Create solid and dashed lines with the same color
for label in file_paths_foot_x.keys():
    if 'ref' in label:
        line, = axFootX.plot([], [], lw=2, linestyle='--', label=label, color=colors_foot_x[label.replace(' ref', '')])
    else:
        line, = axFootX.plot([], [], lw=2, label=label)
        colors_foot_x[label] = line.get_color()
    lines_foot_x[label] = line


# Set up the plot labels and limits
axFootX.set_xlabel('Iteration')
axFootX.set_ylabel('Foot x pos [m]')
axFootX.legend(loc='upper right')
axFootX.set_title('Foot x pos')

axFootX.set_xlim(-0.1, 100.1)
axFootX.set_ylim(-0.35, 0.35)

# ------------- Foot Y --------------
lines_foot_y = {}
colors_foot_y = {}

# Create solid and dashed lines with the same color
for label in file_paths_foot_y.keys():
    if 'ref' in label:
        line, = axFootY.plot([], [], lw=2, linestyle='--', label=label, color=colors_foot_y[label.replace(' ref', '')])
    else:
        line, = axFootY.plot([], [], lw=2, label=label)
        colors_foot_y[label] = line.get_color()
    lines_foot_y[label] = line

# Set up the plot labels and limits
axFootY.set_xlabel('Iteration')
axFootY.set_ylabel('Foot y pos [m]')
axFootY.legend(loc='upper right')
axFootY.set_title('Foot y pos')

axFootY.set_xlim(-0.1, 100.1)
axFootY.set_ylim(0.25, 0.50)

# ------------- Foot Z --------------
lines_foot_z = {}
colors_foot_z = {}
# Create solid and dashed lines with the same color
for label in file_paths_foot_z.keys():
    if 'ref' in label:
        line, = axFootZ.plot([], [], lw=2, linestyle='--', label=label, color=colors_foot_z[label.replace(' ref', '')])
    else:
        line, = axFootZ.plot([], [], lw=2, label=label)
        colors_foot_z[label] = line.get_color()
    lines_foot_z[label] = line

# Set up the plot labels and limits
axFootZ.set_xlabel('Iteration')
axFootZ.set_ylabel('Foot height [m]')
axFootZ.legend(loc='upper right')
axFootZ.set_title('Foot Height')

axFootZ.set_xlim(-0.1, 100.1)
axFootZ.set_ylim(-0.0, 0.10)

count=0
already_init = False

lines_list = [lines_grf, lines_foot_z,lines_height, lines_foot_x, lines_foot_y]
def init():
    for lines in lines_list:
        for line in lines.values():
            line.set_data([], [])


    c = np.loadtxt(winning_policy_path, delimiter=',')
    unique_values = np.unique(c)

    global already_init
    if not already_init :
        already_init = True

        for val in unique_values:
            axCost.scatter([], [], color=plt.cm.viridis(val / float(max(c))), label=f'Value {val}')
            axCost.legend(loc='best')


    return [line for lines in lines_list for line in lines.values()]


def update(frame):
    # ------------- GRF --------------
    try:
        for label, file_path in file_paths_GRF.items():
            if os.path.exists(file_path):
                # Load the data from CSV file
                data = np.loadtxt(file_path, delimiter=',')
                # Select only the third line (row index 2)
                data_to_plot = data[2, :]
                x = np.arange(len(data_to_plot))-1
                y = data_to_plot
                lines_grf[label].set_data(x, y)
                # lines[label].set_ydata(y)
            else:
                print(f"File {file_path} does not exist.")
        
        # Adjust x and y limits based on the data
        all_y_data = [lines_grf[label].get_ydata() for label in file_paths_GRF.keys()]
        if all_y_data:
            y_min = min(-70, min(np.min(y) - 10 for y in all_y_data))
            y_max = max( 70, max(np.max(y) + 10 for y in all_y_data))
            axGRF.set_xlim(-1.2, len(data_to_plot) - 1.8)
            axGRF.set_ylim(y_min, y_max)
    except Exception as e: print(f"Error reading or processing the file: {e}")


    # ---------------- Cost --------------
    try:
        for label, file_path in file_paths_cost.items():
            if os.path.exists(file_path):
                # Load the data from CSV file
                data = np.loadtxt(file_path, delimiter=',')
                # Select only the third line (row index 2)
                data_to_plot = data
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                # lines_cost[label].set_data(x, y)

            else:
                print(f"File {file_path} does not exist.")


        c = np.loadtxt(winning_policy_path, delimiter=',')

        # Split the line into segments
        segments = [[(x[i], y[i]), (x[i+1], y[i+1])] for i in range(len(x)-1)]
        
        # Create an array of colors based on the `c` variable
        colors = plt.cm.viridis(c[:-1] / float(max(c)))

        # # Create a custom legend
        # unique_values = np.unique(c)
        # for val in unique_values:
        #     axCost.scatter([], [], color=plt.cm.viridis(val / float(max(c))), label=f'Value {val}')
        # axCost.legend(loc='best')
        
        # Update the LineCollection
        lines_cost.set_segments(segments)
        lines_cost.set_color(colors)
        
        # Adjust x and y limits based on the data
        y_min = min(0.0,np.min(y) - 20.0 )
        y_max = max(200,np.max(y) + 20.0 )
        axCost.set_xlim(0, len(data_to_plot) + 5.0)
        axCost.set_ylim(y_min, y_max)
        # all_y_data = [lines_cost[label].get_ydata() for label in file_paths_cost.keys()]
        # if all_y_data:
        #     y_min = min(0.0,min(np.min(y) - 20.0 for y in all_y_data))
        #     y_max = max(200,max(np.max(y) + 20.0 for y in all_y_data))
        #     # print(y_max)
        #     axCost.set_xlim(0, len(data_to_plot) + 5.0)
        #     axCost.set_ylim(y_min, y_max)
    except Exception as e: print(f"Error reading or processing the file: {e}")


    # -------------- Height -----------
    try:
        for label, file_path in file_paths_height.items():
            if os.path.exists(file_path):
                # Load the data from CSV file
                data = np.loadtxt(file_path, delimiter=',')
                # Select only the third line (row index 2)
                data_to_plot = data
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                lines_height[label].set_data(x, y)
                # lines[label].set_ydata(y)
            else:
                print(f"File {file_path} does not exist.")
        
        # Adjust x and y limits based on the data
        all_y_data = [lines_height[label].get_ydata() for label in file_paths_height.keys()]
        if all_y_data:
            y_min = min(0.30,min(np.min(y) - 0.03 for y in all_y_data))
            y_max = max(0.40,max(np.max(y) + 0.03 for y in all_y_data))
            axHeight.set_xlim(0, len(data_to_plot) + 5.0)
            axHeight.set_ylim(y_min, y_max)
    except Exception as e: print(f"Error reading or processing the file: {e}")


    # ------------- Foot X --------------
    try:
        for label, file_path in file_paths_foot_x.items():
            if os.path.exists(file_path):
                data = np.loadtxt(file_path, delimiter=',')
                data_to_plot = data[:,0]
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                lines_foot_x[label].set_data(x, y)

            else:
                print(f"File {file_path} does not exist.")
        
        # Adjust x and y limits based on the data
        all_y_data = [lines_foot_x[label].get_ydata() for label in file_paths_foot_x.keys()]
        if all_y_data:
            y_min = min(np.min(y) - 0.03 for y in all_y_data)
            y_max = max(np.max(y) + 0.03 for y in all_y_data)
            axFootX.set_xlim(0, len(data_to_plot) + 5) # Plus 5 to give an impression of live data
            axFootX.set_ylim(y_min, y_max)
    except Exception as e: print(f"Error reading or processing the file: {e}")

    # ------------- Foot Y --------------
    try:
        for label, file_path in file_paths_foot_y.items():
            if os.path.exists(file_path):
                data = np.loadtxt(file_path, delimiter=',')
                data_to_plot = data[:,1]
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                lines_foot_y[label].set_data(x, y)

            else:
                print(f"File {file_path} does not exist.")
        
        # Adjust x and y limits based on the data
        all_y_data = [lines_foot_y[label].get_ydata() for label in file_paths_foot_y.keys()]
        if all_y_data:
            y_min = min(np.min(y) - 0.03 for y in all_y_data)
            y_max = max(np.max(y) + 0.03 for y in all_y_data)
            axFootY.set_xlim(0, len(data_to_plot) + 5) # Plus 5 to give an impression of live data
            axFootY.set_ylim(y_min, y_max)

    except Exception as e: print(f"Error reading or processing the file: {e}")

    # ------------- Foot Z --------------
    try:
        for label, file_path in file_paths_foot_z.items():
            if os.path.exists(file_path):
                data = np.loadtxt(file_path, delimiter=',')
                data_to_plot = data[:,2]
                x = np.arange(len(data_to_plot))
                y = data_to_plot
                lines_foot_z[label].set_data(x, y)

            else:
                print(f"File {file_path} does not exist.")
        
        # Adjust x and y limits based on the data
        all_y_data = [lines_foot_z[label].get_ydata() for label in file_paths_foot_z.keys()]
        if all_y_data:
            y_min = min(-0.0,min(np.min(y) - 0.02 for y in all_y_data))
            y_max = max(0.10,max(np.max(y) + 0.02 for y in all_y_data))
            axFootZ.set_xlim(0, len(data_to_plot) + 5) # Plus 5 to give an impression of live data
            axFootZ.set_ylim(y_min, y_max)
    except Exception as e: print(f"Error reading or processing the file: {e}")

    # Redraw the figure to update the axes
    global count
    if count%40 == 0:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    count+=1

    return [line for lines in lines_list for line in lines.values()]

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    blit=True,
    interval=25,
    cache_frame_data=False,
)

plt.show()

