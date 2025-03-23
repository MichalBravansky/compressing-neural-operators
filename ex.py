import matplotlib.pyplot as plt
import numpy as np

# Experimental results from eval_all_models.py:
models = ['FNO', 'DeepONet']

# Baseline and pruned L2 losses:
baseline_l2 = [0.18928355191435134, 0.061397962272167206]
pruned_l2   = [0.34888674105916706, 0.10647908811058317]

# Baseline and pruned H1 losses:
baseline_h1 = [1.256005849157061, 0.3458086635385241]
pruned_h1   = [1.2533196551459176, 0.49273700373513357]

x = np.arange(len(models))  # label locations
width = 0.35  # width of the bars

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot for L2 loss:
rects1 = ax1.bar(x - width/2, baseline_l2, width, label='Baseline')
rects2 = ax1.bar(x + width/2, pruned_l2, width, label='Pruned')
ax1.set_ylabel('L2 Loss')
ax1.set_title('Comparison of L2 Loss')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot for H1 loss:
rects3 = ax2.bar(x - width/2, baseline_h1, width, label='Baseline')
rects4 = ax2.bar(x + width/2, pruned_h1, width, label='Pruned')
ax2.set_ylabel('H1 Loss')
ax2.set_title('Comparison of H1 Loss')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

fig.tight_layout()
plt.savefig('LayerPruning_LossComparison.pdf')
plt.show()
