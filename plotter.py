import matplotlib.pyplot as plt

# Data
data = {
    'coreset': [0.3944, 0.59095, 0.64035, 0.6685, 0.69405, 0.71955],
    'vaal': [0.3666, 0.5066, 0.6198, 0.67535, 0.7187, 0.7473],
    'tavaal': [0.3933, 0.5086, 0.60195, 0.6795, 0.6892, 0.7171],
    'sraal': [0.3713, 0.51835, 0.61325, 0.64565, 0.67705, 0.69605],
    'our method': [0.4025, 0.5608, 0.6763, 0.71605, 0.74725, 0.7688]
}


# Multiply each value by 100
for method, values in data.items():
    data[method] = [val * 100 for val in values]

# Plot
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor('lightgrey')  # Set background color to grey

# Define markers
markers = ['o', 's', '^', 'D', 'x']

# Plot each line with different marker
for i, (method, values) in enumerate(data.items()):
    plt.plot(range(1, len(values) + 1), values, marker=markers[i % len(markers)], label=method,linewidth=2)

plt.title('WEBNLG')
plt.xlabel('Training Samples')
plt.ylabel('Mean Accuracy(%)')
plt.legend()
plt.grid(True)
plt.xticks(range(1, len(data['coreset']) + 1))
plt.xticks(range(1, len(next(iter(data.values()))) + 1), [str(170 + i * 30) for i in range(len(next(iter(data.values()))))])
plt.savefig('/mnt/d/BITS_Acads/project/plots_al/training_progress_WEBNLG_9cycles_final.png')

# Show plot
plt.show()