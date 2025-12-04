import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# DISCLAIMER: This file was generated using LLM

def save_benchmarks():
    # 1. Benchmark Data
    data = {
        'Cluster': ['Cluster 1', 'Cluster 1', 'Cluster 2', 'Cluster 2', 'Cluster 3', 'Cluster 3'],
        'Strategy': ['RANDOM', 'HYBRID', 'RANDOM', 'HYBRID', 'RANDOM', 'HYBRID'],
        'Mean': [5700717.16, 5350679.36, 9843927.86, 9207704.20, 16739128.88, 15824350.75],
        'Min': [5319776.44, 4490162.97, 8753402.21, 7157535.51, 14666891.45, 13608586.16]
    }

    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")

    def format_millions(x, pos):
        return f'{x * 1e-6:.1f}M'

    def annotate_improvement(ax, metric_col):
        files = df['Cluster'].unique()
        for i, f in enumerate(files):
            val_random = df[(df['Cluster'] == f) & (df['Strategy'] == 'RANDOM')][metric_col].values[0]
            val_hybrid = df[(df['Cluster'] == f) & (df['Strategy'] == 'HYBRID')][metric_col].values[0]

            reduction = ((val_random - val_hybrid) / val_random) * 100

            x_pos = i + 0.20
            y_pos = val_hybrid + (val_hybrid * 0.02)

            ax.text(x_pos, y_pos, f'-{reduction:.1f}%', ha='center', va='bottom',
                    fontweight='bold', color='#c0392b', fontsize=12)

    # --- FIGURE 1: Mean Fitness (Consistency) ---
    plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(data=df, x='Cluster', y='Mean', hue='Strategy', palette=['#95a5a6', '#e74c3c'])

    ax1.set_title('Fitness Médio', fontsize=14, pad=15)
    ax1.set_ylabel('Fitness Score (Menor é Melhor)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_millions))
    annotate_improvement(ax1, 'Mean')

    plt.tight_layout()
    plt.savefig('benchmark_mean_consistency.png', dpi=300)
    print("Saved: benchmark_mean_consistency.png")
    plt.close()  # Close to free memory

    # --- FIGURE 2: Min Fitness (Peak Performance) ---
    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(data=df, x='Cluster', y='Min', hue='Strategy', palette=['#95a5a6', '#e74c3c'])

    ax2.set_title('Melhor Fitness Encontrado', fontsize=14, pad=15)
    ax2.set_ylabel('Fitness Score (Menor é Melhor)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_millions))
    annotate_improvement(ax2, 'Min')

    plt.tight_layout()
    plt.savefig('benchmark_min_peak.png', dpi=300)
    print("Saved: benchmark_min_peak.png")
    plt.close()