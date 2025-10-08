import json
import matplotlib.pyplot as plt
from scipy import stats

def evaluate_threshold(major_scores, minor_scores, alpha=0.05, save_dir="results"):
    if len(major_scores) < 2 or len(minor_scores) < 2:
        print("❌ Not enough data points for t-test.")
        return

    t_stat, p_value = stats.ttest_ind(major_scores, minor_scores, equal_var=False)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    interpretation = (
        "✅ Significant difference. Threshold is valid."
        if p_value < alpha else
        "⚠️ No significant difference. Adjust threshold."
    )
    print(interpretation)

    # save numeric results
    with open(f"{save_dir}/metrics/threshold_eval.json", "w") as f:
        json.dump({"t_stat": t_stat, "p_value": p_value, "alpha": alpha}, f, indent=2)

    # histograms
    plt.figure(figsize=(10, 5))
    plt.hist(major_scores, bins=20, alpha=0.7, label='Major Damage', color='red')
    plt.hist(minor_scores, bins=20, alpha=0.7, label='Minor Damage', color='blue')
    plt.xlabel('Squiggliness Score'); plt.ylabel('Frequency')
    plt.legend(); plt.title('Histogram of Squiggliness Scores')
    plt.savefig(f"{save_dir}/figures/histogram.png")

    # boxplot
    plt.figure(figsize=(7, 5))
    plt.boxplot([major_scores, minor_scores], labels=['Major', 'Minor'])
    plt.ylabel('Squiggliness Score')
    plt.title('Boxplot of Squiggliness Scores')
    plt.savefig(f"{save_dir}/figures/boxplot.png")

    print("📊 Results saved to results/metrics and results/figures.")
