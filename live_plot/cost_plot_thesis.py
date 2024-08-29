import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, json, torch
import pandas as pd

file_path = 'live_variable/cost.csv'

style_theme = 'whitegrid'
sns.set_theme(style=style_theme)
fig, ax = plt.subplots(figsize=(4.5, 4.5)) #(10,7)
fontsize=14

values = np.loadtxt('live_variable/cost.csv', delimiter=',')
labels = np.loadtxt('live_variable/winning_policy.csv', delimiter=',', dtype=int)
# label_dict = {0: 'Base warm Start', 1:'Speed warm start'}
label_dict2 = {1: 'Base warm Start', 0:'Speed warm start'}

# cmap = plt.get_cmap('viridis', 2) 
cmap = plt.get_cmap('tab10')
# cmap = plt.get_cmap('Set1')


colors = cmap(labels)

for i in range(len(values) - 1):
    plt.plot([i/100,(i+1)/100], values[i:i+2], color=colors[i], linewidth=2.5)

unique_labels = np.unique(labels)
for label in unique_labels:
    plt.plot([], [], color=cmap(label), label=f'{label_dict2[label]}')


ax.set_xlabel('Time [s]',fontsize=fontsize)
# ax.set_ylabel('Best trajectory cost')
ax.set_title('Rollout Cost at 0.1m/s',fontsize=fontsize)
ax.legend(loc='upper left',fontsize=fontsize)

ax.set_xlim(0, 1)
# ax.set_ylim(0.25, 0.50)

fig.tight_layout()
plt.savefig('base_speed_multipolcy_cost_01ms.pdf')
plt.show()