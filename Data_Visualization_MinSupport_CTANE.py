import matplotlib.pyplot as plt

# Data points
x = [15, 25, 50]
y_CTANE = [0.37, 0.69, 0.88]

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(x, y_CTANE, marker='o', linestyle='-', label='CTANE')


# Add labels and title
plt.xlabel('Min. Support (%)')
plt.ylabel('Time (s)')
plt.title('Nursery')

# Add legend
plt.legend(loc='upper left')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
