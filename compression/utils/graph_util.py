import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def generate_graph(results: dict, hyperparameters: dict, model_name: str, measureable: str, unit: str, savefile: str = None):
    metrics = ["l2_loss_increase", "model_size_reduction", 
               "peak_memory_reduction", "flops_reduction"]  
    fig = plt.figure(figsize=(15, 10)) 
    fig.suptitle(f'Comparison of {model_name.replace("_"," ").title()} Compression Performance', fontsize=16)

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])  

    markers = ['o', 's', 'D', '^', 'v', 'p', '*']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f1c40f', '#ff5733', '#1abc9c'] 

    plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  
    axes = []  

    global_handles = [] 
    global_labels = []  

    for i, metric in enumerate(metrics):
        row, col = plot_positions[i]  
        ax = fig.add_subplot(gs[row, col]) 
        axes.append(ax)

        for j, model in enumerate(results.keys()):
            if model not in hyperparameters:
                continue  
            
            x_values = hyperparameters[model] 
            y_values = [
                results[model].get("Comparison", {}).get(ratio, {}).get(metric, None) 
                for ratio in x_values
            ]

            x_filtered = [x for x, y in zip(x_values, y_values) if y is not None]
            y_filtered = [y for y in y_values if y is not None]

            if not x_filtered or not y_filtered:
                print(f"Warning: No data found for {model} at {metric}")
                continue  

            line, = ax.plot(np.array(x_filtered) * 100, y_filtered,
                            f'--{markers[j]}', label=f'{model}',
                            color=colors[j % len(colors)], alpha=0.7)

            if i == 0 and line not in global_handles:
                global_handles.append(line)
                global_labels.append(model)

        ax.set_xlabel(f'{model_name.replace("_"," ").title()} {measureable} ({unit})')
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} Ratio (%)")
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
        ax.grid(True, linestyle='--', alpha=0.7)

    if global_handles:
        fig.legend(handles=global_handles, labels=global_labels, 
                   loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=len(global_labels),
                   fontsize=12, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.88]) 
    
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'compression/utils/all_{model_name}_performance.png', dpi=300, bbox_inches='tight')
    
    plt.show()



