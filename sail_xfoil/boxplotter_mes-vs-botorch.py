import re
import matplotlib.pyplot as plt

# Extracting Ratio values
ratio_values = []
with open("mes-vs-botorch-init.log", "r") as file:
    for line in file:
        match = re.search(r"Ratio: (\d+\.\d+)", line)
        if match:
            ratio_values.append(float(match.group(1)))

ratio_smaller_1 = [x for x in ratio_values if x < 1]
ratio_greater_1 = [x for x in ratio_values if x > 1]
mean_ratio_smaller_1 = sum(ratio_smaller_1) / len(ratio_smaller_1)
mean_ratio_greater_1 = sum(ratio_greater_1) / len(ratio_greater_1)

# Creating boxplot
plt.boxplot(ratio_smaller_1)
plt.text(1.1, 0.9, f"n={len(ratio_smaller_1)}", fontsize=12) # print number of values in top right corner
plt.text(1.1, 0.8, f"mean={mean_ratio_smaller_1:.2f}", fontsize=12)
plt.title('Ratio Smaller 1')
plt.ylabel('Ratio Values')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.axhline(y=50, color='r', linestyle='--')  # Draw median line in red
plt.axhline(y=mean_ratio_smaller_1, color='g', linestyle='--')
plt.savefig('imgs/init_ratio_smaller_1.png')

plt.clf() # clear boxplot settings

# Creating new boxplot
plt.boxplot(ratio_greater_1)
plt.text(1.1, 9, f"n={len(ratio_greater_1)}", fontsize=12) # print number of values in top right corner
plt.text(1.1, 8, f"mean={mean_ratio_greater_1:.2f}", fontsize=12)
plt.title('Ratio Greater 1')
plt.ylabel('Ratio Values')
plt.ylim(1, 10)  # Set y-axis scale from 1 to 2
plt.axhline(y=50, color='r', linestyle='--')  # Draw median line in red
plt.axhline(y=mean_ratio_greater_1, color='g', linestyle='--')
plt.savefig('imgs/init_ratio_greater_1.png')

plt.clf() # clear boxplot settings

with open("mes-vs-botorch-obj-evals.log", "r") as file:
    for line in file:
        match = re.search(r"Ratio: (\d+\.\d+)", line)
        if match:
            ratio_values.append(float(match.group(1)))

ratio_smaller_1 = [x for x in ratio_values if x < 1]
ratio_greater_1 = [x for x in ratio_values if x > 1]
mean_ratio_smaller_1 = sum(ratio_smaller_1) / len(ratio_smaller_1)
mean_ratio_greater_1 = sum(ratio_greater_1) / len(ratio_greater_1)

# Creating boxplot
plt.boxplot(ratio_smaller_1)
plt.text(1.1, 0.9, f"n={len(ratio_smaller_1)}", fontsize=12) # print number of values in top right corner
plt.text(1.1, 0.8, f"mean={mean_ratio_smaller_1:.2f}", fontsize=12)
plt.title('Ratio Smaller 1')
plt.ylabel('Ratio Values')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.axhline(y=50, color='r', linestyle='--')  # Draw median line in red
plt.axhline(y=mean_ratio_smaller_1, color='g', linestyle='--')
plt.savefig('imgs/final_ratio_smaller_1.png')

plt.clf() # clear boxplot settings

# Creating new boxplot
plt.boxplot(ratio_greater_1)
plt.text(1.1, 9, f"n={len(ratio_greater_1)}", fontsize=12) # print number of values in top right corner
plt.text(1.1, 8, f"mean={mean_ratio_greater_1:.2f}", fontsize=12)
plt.title('Ratio Greater 1')
plt.ylabel('Ratio Values')
plt.ylim(1, 10)  # Set y-axis scale from 1 to 2
plt.axhline(y=50, color='r', linestyle='--')  # Draw median line in red
plt.axhline(y=mean_ratio_greater_1, color='g', linestyle='--')
plt.savefig('imgs/final_ratio_greater_1.png')
