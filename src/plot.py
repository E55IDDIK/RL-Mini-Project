import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves():
    # Load the re-evaluated data which has 5 rows per checkpoint (one for each eval seed)
    df = pd.read_csv("results/learning_curve_data.csv")
    
    # Set theme
    sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)
    
    # Plot Return
    plt.figure(figsize=(8, 5))
    
    # sns.lineplot automatically computes the mean and a confidence interval (or standard deviation) 
    # across the multiple seeds for each step. We use errorbar="sd" to show standard deviation.
    sns.lineplot(data=df, x="step", y="episode_return", hue="method", 
                 errorbar="sd", linewidth=2, palette=["blue", "orange"])

    plt.title("Learning Curve: Episode Return")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Return (5 Eval Seeds)")
    
    # Clean up legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [label.upper().replace("_", "-") for label in labels]
    plt.legend(handles, labels)
    
    plt.tight_layout()
    plt.savefig("figures/learning_curve_return.pdf", bbox_inches="tight")
    plt.close()

def plot_constraints():
    df = pd.read_csv("results/learning_curve_data.csv")
    
    sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cost
    sns.lineplot(data=df, x="step", y="total_cost", hue="method", 
                 errorbar="sd", ax=axes[0], palette=["blue", "orange"])
    axes[0].set_title("Travel Cost")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Mean Cost")
    
    # Clean up legend labels for cost plot
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [label.upper().replace("_", "-") for label in labels]
    axes[0].legend(handles, labels)
    
    # Late
    sns.lineplot(data=df, x="step", y="n_late", hue="method", 
                 errorbar="sd", ax=axes[1], palette=["blue", "orange"])
    axes[1].set_title("Late Deliveries")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Mean Late")
    
    # Clean up legend labels for late plot
    handles, labels = axes[1].get_legend_handles_labels()
    labels = [label.upper().replace("_", "-") for label in labels]
    axes[1].legend(handles, labels)
    
    plt.tight_layout()
    plt.savefig("figures/learning_curve_metrics.pdf", bbox_inches="tight")
    plt.close()

def plot_benchmark():
    df = pd.read_csv("results/eval_comparison.csv")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Return
    sns.barplot(data=df, x="method", y="episode_return", ax=axes[0], errorbar="sd", palette="viridis", hue="method", legend=False)
    axes[0].set_title("Final Episode Return")
    axes[0].set_ylabel("Return")
    
    # 2. Cost
    sns.barplot(data=df, x="method", y="total_cost", ax=axes[1], errorbar="sd", palette="viridis", hue="method", legend=False)
    axes[1].set_title("Total Travel Cost")
    axes[1].set_ylabel("Cost")
    
    # 3. On Time
    sns.barplot(data=df, x="method", y="n_on_time", ax=axes[2], errorbar="sd", palette="viridis", hue="method", legend=False)
    axes[2].set_title("On-Time Deliveries")
    axes[2].set_ylabel("Count")
    
    for ax in axes:
        ax.set_xlabel("")
        labels = [item.get_text().upper().replace("_", "-") for item in ax.get_xticklabels()]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels, rotation=30)
        
    plt.tight_layout()
    plt.savefig("figures/benchmark_comparison.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_learning_curves()
    plot_constraints()
    plot_benchmark()
    print("Generated figures in figures/")
