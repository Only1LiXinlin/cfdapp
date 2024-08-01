import matplotlib.pyplot as plt

# Data points
x = [10, 25, 50]
y_CFDMine = [610.73, 2549.39, 7365.42]

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(x, y_CFDMine, marker='s', linestyle='-', label='CFD-Mine', color = 'green')


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
