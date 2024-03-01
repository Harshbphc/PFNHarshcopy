import matplotlib.pyplot as plt

# Data
data = {
    'coreset': [0.00625, 0, 0, 0.3862, 0.60075, 0.6405, 0.65975],
    'vaal': [0.00625, 0, 0, 0, 0.40455, 0.5748, 0.66825],
    'tavaal': [0.00355, 0, 0, 0, 0.40595, 0.57185, 0.65055],
    'sraal': [0.00625, 0, 0, 0, 0.40585, 0.5545, 0.66305],
    'our method': [0.00625, 0, 0, 0, 0.6078, 0.6912, 0.72415]
}


# Multiply each value by 100
for method, values in data.items():
    data[method] = [val * 100 for val in values]

# Plot
plt.figure(figsize=(10, 6))

# Define markers
markers = ['o', 's', '^', 'D', 'x']

# Plot each line with different marker
for i, (method, values) in enumerate(data.items()):
    plt.plot(range(1, len(values) + 1), values, marker=markers[i % len(markers)], label=method)

plt.title('Training Progress')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, len(data['coreset']) + 1))

plt.savefig('/mnt/d/BITS_Acads/project/plots_al/training_progress_webnlg.png')

# Show plot
plt.show()