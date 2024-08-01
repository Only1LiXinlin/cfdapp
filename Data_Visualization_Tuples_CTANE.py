import matplotlib.pyplot as plt

# Data points
x = [10, 25, 50]
y_CTANE = [2.05, 3.11, 5.24]

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(x, y_CTANE, marker='o', linestyle='-', label='CTANE')


# Add labels and title
plt.xlabel('% Tuples')
plt.ylabel('Time (s)')
plt.title('Nursery')

# Add legend
plt.legend(loc='upper left')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
