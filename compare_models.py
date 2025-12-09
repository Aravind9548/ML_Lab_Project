import matplotlib.pyplot as plt
import numpy as np

def main():
    # --- Data Definitions ---
    models = ["117M", "345M", "762M", "1542M"]
    x_pos = np.arange(len(models))

    # Paper Data (Baseline - from plot_fake_metrics.py)
    paper_data = {
        "WikiText-2": [29.41, 22.76, 19.93, 18.34], # Lower is better
        "LAMBADA":    [45.99, 55.48, 60.12, 63.24], # Higher is better
        "Winograd":   [52.00, 62.00, 68.00, 70.70], # Higher is better
        "CoQA":       [25.00, 44.00, 50.00, 55.00], # Higher is better
        "Summarization": [20.00, 23.00, 24.50, 26.58], # Higher is better
        "Translation":   [6.00, 8.00, 10.00, 11.50]   # Higher is better
    }

    # Our Data (Fine-Tuned - Synthesized based on user request)
    # "Our" is generally improved (Fine-Tuned), but worse in some cases.
    our_data = {
        "WikiText-2": [25.10, 19.50, 16.80, 15.20], # Improved (Lower)
        "LAMBADA":    [48.50, 58.20, 63.50, 66.80], # Improved (Higher)
        "Winograd":   [54.00, 64.50, 70.00, 72.50], # Improved (Higher)
        "CoQA":       [22.00, 40.00, 48.00, 52.00], # WORSE (Regression/Overfitting?)
        "Summarization": [22.50, 25.00, 27.00, 29.00], # Improved (Higher)
        "Translation":   [5.00, 7.00, 9.50, 10.80]     # WORSE (Forgetting)
    }

    metrics_info = {
        "WikiText-2": {"title": "WikiText-2 (Language Modeling)", "ylabel": "Perplexity (Lower is Better)", "better": "lower"},
        "LAMBADA":    {"title": "LAMBADA (Long-Range Dependencies)", "ylabel": "Accuracy % (Higher is Better)", "better": "higher"},
        "Winograd":   {"title": "Winograd Schema (Reasoning)", "ylabel": "Accuracy % (Higher is Better)", "better": "higher"},
        "CoQA":       {"title": "CoQA (Reading Comprehension)", "ylabel": "F1 Score (Higher is Better)", "better": "higher"},
        "Summarization": {"title": "Summarization (CNN/Daily Mail)", "ylabel": "ROUGE-L (Higher is Better)", "better": "higher"},
        "Translation":   {"title": "Translation (French-to-English)", "ylabel": "BLEU (Higher is Better)", "better": "higher"}
    }

    # --- Plotting ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Model Comparison: Paper vs. Ours (Fine-Tuned)', fontsize=20, fontweight='bold')
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    for i, (task_name, p_values) in enumerate(paper_data.items()):
        ax = axes_flat[i]
        o_values = our_data[task_name]
        info = metrics_info[task_name]

        # Plot Paper Line
        ax.plot(models, p_values, marker='o', linestyle='-', color='gray', label='Paper (Baseline)', linewidth=2)
        
        # Plot Our Line
        ax.plot(models, o_values, marker='s', linestyle='--', color='blue', label='Ours (Fine-Tuned)', linewidth=2)

        # Annotate Our Values (Optional, to show specific numbers)
        for j, val in enumerate(o_values):
             ax.text(j, val, f"{val}", ha='center', va='bottom', fontsize=9, color='blue')

        ax.set_title(info["title"], fontsize=14, fontweight='bold')
        ax.set_xlabel("Model Size")
        ax.set_ylabel(info["ylabel"])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    output_file = "comparison_plots.png"
    plt.savefig(output_file)
    print(f"Comparison plots saved to {output_file}")
    plt.show() # Optional: Show if environment supports it

if __name__ == "__main__":
    main()
