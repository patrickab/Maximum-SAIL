import re
import matplotlib.pyplot as plt

# Extracting Ratio values
ratio_values = []
with open("mes-vs-botorch-init.log", "r") as file:
    for line in file:
        match = re.search(r"Ratio: (\d+\.\d+)", line)
        if match:
            ratio_values.append(float(match.group(1)))

# Creating boxplot
plt.boxplot(ratio_values)
plt.title('Ratio Boxplot')
plt.ylabel('Ratio Values')
plt.ylim(0, 10)  # Set y-axis scale from 0 to 100
plt.axhline(y=50, color='r', linestyle='--')  # Draw median line in red
plt.axhline(y=sum(ratio_values) / len(ratio_values), color='g', linestyle='--')
plt.show()
