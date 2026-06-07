import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Data
    labels = ['Right Ventricle (RV)', 'Myocardium (MYO)', 'Left Ventricle (LV)']
    baseline = [83.12, 74.48, 88.46]
    hr_std = [84.99, 74.34, 87.15]
    hr_cca = [86.44, 75.46, 90.11]

    # SOTA Ranges: (low, high)
    sota_ranges = [
        (81.0, 88.0), # RV
        (84.0, 88.0), # MYO
        (93.0, 96.0)  # LV
    ]

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Draw SOTA Range bands first so bars overlay them
    for idx, (low, high) in enumerate(sota_ranges):
        ax.fill_between(
            [idx - 0.35, idx + 0.35], 
            low, 
            high, 
            color='forestgreen', 
            alpha=0.12, 
            label='Global SOTA Benchmark Range' if idx == 0 else ""
        )
        ax.hlines(high, idx - 0.35, idx + 0.35, colors='forestgreen', linestyles='dashed', alpha=0.5)
        ax.hlines(low, idx - 0.35, idx + 0.35, colors='forestgreen', linestyles='dashed', alpha=0.5)
        # Add text label for SOTA range
        ax.text(idx, high + 0.8, f"SOTA: {low}%~{high}%", ha='center', color='forestgreen', fontsize=9, fontweight='bold')

    # Bars
    rects1 = ax.bar(x - width, baseline, width, label='Baseline Model (Low-Res)', color='#b0bec5')
    rects2 = ax.bar(x, hr_std, width, label='HR Model (Standard)', color='#64b5f6')
    rects3 = ax.bar(x + width, hr_cca, width, label='Final Model (HR + CCA)', color='#1e88e5')

    # Labels and styling
    ax.set_ylabel('Dice Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('CardioSeg3D Performance vs. MICCAI ACDC Global Benchmarks', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(60, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Legend
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    chart_path = 'assets/benchmark_comparison_chart.png'
    plt.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Saved benchmark comparison chart to {chart_path}")

if __name__ == "__main__":
    main()
