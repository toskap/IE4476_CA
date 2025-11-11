import matplotlib.pyplot as plt
import tkinter as tk


components = ['No PCA (12288)', 'PCA 10', 'PCA 50', 'PCA 100', 'PCA 200', 'PCA 300', 'PCA 400', 'PCA 500', 'PCA 600', 'PCA 1000']
accuracies = [0.3939, 0.3382, 0.3767, 0.3976, 0.3998, 0.3972, 0.3968, 0.3965, 0.3928, 0.3957]


plt.figure(figsize=(9,8))
bars = plt.bar(components, accuracies, color='orange', alpha=0.8)
bars[0].set_color('#cd3700')
plt.axhline(y=accuracies[0], color='#cd3700', linestyle='--', linewidth=1.5)

plt.title("Classification Accuracy vs number of PCA Components", fontsize=12)
plt.xlabel("Number of PCA Components", fontsize=11)
plt.ylabel("Test Accuracy", fontsize=11)
plt.xticks(rotation=45, ha="right")
plt.ylim(0.32, 0.41)
plt.grid(axis='y', linestyle='--', alpha=0.6)


for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.0003, f"{acc:.4f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
