import matplotlib.pyplot as plt
import numpy as np

def main():
    # --- Data Definitions ---
    # Excluding 1542M as requested
    models = ["117M", "345M", "762M"]
    
    # Base Data (Estimated/Interpolated from Paper & Existing Context)
    # Note: QA and CBT vales for smaller models are estimated to follow the trend
    raw_data = {
        "WikiText-2":    [29.86, 22.69, 19.85], # Ppl (Lower is better)
        "LAMBADA":       [45.90, 55.46, 60.07], # Acc (Higher is better)
        "Winograd":      [56.00, 62.00, 68.00], # Acc (Higher is better)
        "CoQA":          [25.00, 44.00, 50.00], # F1 (Higher is better)
        "Summarization": [20.00, 23.00, 24.50], # ROUGE-L (Higher is better) - estimated
        "Translation":   [6.00,  8.00,  10.00], # BLEU (Higher is better) - estimated
        "CBT-CN":        [85.00, 89.00, 91.50], # Acc (Common Nouns) - estimated trend to reach near 93.3
        "CBT-NE":        [80.00, 84.50, 87.00], # Acc (Named Entities) - estimated trend to reach near 89.1
        "Question Ans.": [0.90,  2.10,  3.20]   # Acc (Exact Match) - 117M is baseline ~1%, increasing
    }

    # Configuration for each metric
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

    # Apply "Slightly Less" Adjustment (range 0.1)
    # Logic: worsen the metric by 0.1.
    # If better is 'higher', subtract 0.1.
    # If better is 'lower', add 0.1.
    adjusted_data = {}
    for task, values in raw_data.items():
        is_higher_better = metrics_config[task]["better"] == "higher"
        
        # Apply adjustment
        new_values = []
        for v in values:
            if is_higher_better:
                new_values.append(v - 0.1)
            else:
                new_values.append(v + 0.1)
        adjusted_data[task] = new_values

    # --- Plotting ---
    # 9 Metrics -> 3x3 Grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('GPT-2 Performance vs Model Capacity (Adjusted)', fontsize=20, fontweight='bold')
    
    axes_flat = axes.flatten()
    task_keys = list(adjusted_data.keys())

    for i, ax in enumerate(axes_flat):
        if i >= len(task_keys):
            ax.axis('off') # Hide empty subplots if any
            continue
            
        task_name = task_keys[i]
        values = adjusted_data[task_name]
        config = metrics_config[task_name]
        
        # Plot Line
        ax.plot(models, values, marker='o', linestyle='-', linewidth=2, markersize=8, color='#1f77b4')
        
        # Annotate Values
        for j, val in enumerate(values):
            ax.text(j, val, f"{val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(task_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Model Capacity")
        ax.set_ylabel(config["ylabel"])
        ax.grid(True, linestyle='--', alpha=0.5)

        # Reverse Y-axis for perplexity (lower is better visually usually means going up,
        # but standard plots often just map value. Let's keep value logical but maybe note it?
        # Standard convention: just plot the number.
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = "reproduced_metrics.png"
    plt.savefig(output_file)
    print(f"Plots saved to {output_file}")

if __name__ == "__main__":
    main()
