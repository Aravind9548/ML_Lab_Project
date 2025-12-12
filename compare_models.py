import matplotlib.pyplot as plt
import numpy as np

def main():
    models = ["117M", "345M", "762M"]
    
    raw_data = {
        "WikiText-2":    [29.86, 22.69, 19.85], 
        "LAMBADA":       [45.90, 55.46, 60.07], 
        "Winograd":      [63.05, 63.1, 69.03], 
        "CoQA":          [25.43, 44.54, 50.67], 
        "Summarization": [20.43, 23.32, 24.59], 
        "Translation":   [6.03,  8.67,  10.78], 
        "CBT-CN":        [87.47, 91.54, 92.57], 
        "CBT-NE":        [84.23, 86.7, 86.2], 
        "Question Ans.": [0.90,  2.1,  3.2]   
    }

    metrics_config = {
        "WikiText-2":    {"ylabel": "Perplexity", "better": "lower"},
        "LAMBADA":       {"ylabel": "Accuracy (%)", "better": "higher"},
        "Winograd":      {"ylabel": "Accuracy (%)", "better": "higher"},
        "CoQA":          {"ylabel": "F1 Score", "better": "higher"},
        "Summarization": {"ylabel": "ROUGE-L", "better": "higher"},
        "Translation":   {"ylabel": "BLEU", "better": "higher"},
        "CBT-CN":        {"ylabel": "Accuracy (%)", "better": "higher"},
        "CBT-NE":        {"ylabel": "Accuracy (%)", "better": "higher"},
        "Question Ans.": {"ylabel": "Accuracy (%)", "better": "higher"}
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('GPT-2 Performance vs Model Capacity', fontsize=20, fontweight='bold')
    
    axes_flat = axes.flatten()
    task_keys = list(raw_data.keys())

    for i, ax in enumerate(axes_flat):
        if i >= len(task_keys):
            ax.axis('off') 
            continue
            
        task_name = task_keys[i]
        values = raw_data[task_name]
        config = metrics_config[task_name]
        
        ax.plot(models, values, marker='o', linestyle='-', linewidth=2, markersize=8, color='#1f77b4')
        
        for j, val in enumerate(values):
            ax.text(j, val, f"{val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(task_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Model Capacity")
        ax.set_ylabel(config["ylabel"])
        ax.grid(True, linestyle='--', alpha=0.5)

        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = "reproduced_metrics.png"
    plt.savefig(output_file)
    print(f"Plots saved to {output_file}")

if __name__ == "__main__":
    main()
