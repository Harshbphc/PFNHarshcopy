import matplotlib.pyplot as plt

# Generating x values starting from 110 and increasing by 30
x_values = [110 + i*30 for i in range(10)]  # generating 10 points

# Generating corresponding y values (for example, y = x)
y_values = [x for x in x_values]

# Plotting the graph
plt.plot(x_values, y_values, marker='o')  # 'o' for points
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph with X values starting from 110 and increasing by 30')
plt.grid(True)
plt.show()