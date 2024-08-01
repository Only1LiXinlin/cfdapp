import matplotlib.pyplot as plt

# Data points
x = [15, 25, 50]
y_CFDMine = [8.76, 8.89, 8.82]

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(x, y_CFDMine, marker='s', linestyle='-', label='CFD-Mine', color = 'green')


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
