import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FILE = "model_comparison_762M.png"

models = ["117M", "345M", "762M"]

tasks = {
    "WikiText-2 (Language Modeling)": {
        "metric": "Perplexity (Lower is Better)",
        "paper_baseline": [29.41, 22.76, 19.93],
        "ours_finetuned": [25.10, 19.50, 16.80],  
        "better_direction": "lower"
    },
    "LAMBADA (Long-Range Dependencies)": {
        "metric": "Accuracy % (Higher is Better)",
        "paper_baseline": [45.99, 55.48, 60.12],
        "ours_finetuned": [48.50, 58.20, 63.50],  
        "better_direction": "higher"
    },
    "Winograd Schema (Reasoning)": {
        "metric": "Accuracy % (Higher is Better)",
        "paper_baseline": [52.00, 62.00, 68.00],
        "ours_finetuned": [54.00, 64.50, 70.10],  
        "better_direction": "higher"
    },
    "CoQA (Reading Comprehension)": {
        "metric": "F1 Score (Higher is Better)",
        "paper_baseline": [25.00, 44.00, 50.00],
        "ours_finetuned": [22.50, 40.10, 48.00],  
        "better_direction": "higher"
    },
    "Summarization (CNN/Daily Mail)": {
        "metric": "ROUGE-L (Higher is Better)",
        "paper_baseline": [20.00, 23.00, 24.50],
        "ours_finetuned": [22.50, 25.10, 27.00],  
        "better_direction": "higher"
    },
    "Translation (French-to-English)": {
        "metric": "BLEU (Higher is Better)",
        "paper_baseline": [6.00, 8.00, 10.00],
        "ours_finetuned": [5.00, 7.00, 9.50],     
        "better_direction": "higher"
    }
}

def plot_truncated_grid():
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("Model Comparison: Paper vs. Ours (Fine-Tuned)", fontsize=20, fontweight='bold', y=0.96)
    
    axes = axes.flatten()
    
    for i, (task_name, task_data) in enumerate(tasks.items()):
        ax = axes[i]
        
        y_paper = task_data["paper_baseline"]
        y_ours = task_data["ours_finetuned"]
        
        ax.plot(models, y_paper, marker='o', linestyle='-', color='grey', linewidth=2, label='Paper (Baseline)')
        
        ax.plot(models, y_ours, marker='D', linestyle='--', color='blue', linewidth=2, label='Ours (Fine-Tuned)')
        
        for j, val in enumerate(y_paper):
            ax.text(j, val, f"{val}", ha='center', va='bottom', fontsize=8, color='black')
        for j, val in enumerate(y_ours):
            offset = 10 if task_data["better_direction"] == "higher" and val > y_paper[j] else -15
            ax.text(j, val, f"{val}", ha='center', va='top' if offset < 0 else 'bottom', fontsize=8, color='blue', fontweight='bold')

        ax.set_title(task_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(task_data["metric"], fontsize=9)
        ax.set_xlabel("Model Size", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Graph saved as {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    plot_truncated_grid()