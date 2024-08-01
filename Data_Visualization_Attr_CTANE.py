import matplotlib.pyplot as plt

# Data points
x = [4, 6, 8]
y_CTANE = [0.27, 0.88, 1.36]

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(x, y_CTANE, marker='o', linestyle='-', label='CTANE')


# Add labels and title
plt.xlabel('Nr. Attributes')
plt.ylabel('Time (s)')
plt.title('Nursery')

# Add legend
plt.legend(loc='upper left')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
