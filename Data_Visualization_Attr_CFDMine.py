import matplotlib.pyplot as plt

# Data points
x = [4, 6, 8]
y_CFDMine = [1.88, 28.71, 595.00]

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(x, y_CFDMine, marker='s', linestyle='-', label='CFD-Mine', color = 'green')


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
