import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# load all_results from pickle at "vit_results.pkl"
with open("vit_results.pkl", "rb") as f:
    all_results = pickle.load(f)

steps_per_epoch = len(train_loader)


def smooth(values, window=50):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# Plot 1 — Smoothed train loss per step
for kind, runs in all_results.items():
    losses = np.array([smooth(r["train_losses"]) for r in runs])
    mean, std = losses.mean(0), losses.std(0)
    axes[0].plot(mean, label=kind)
    axes[0].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

# Add epoch boundaries
for e in range(1, cfg.epochs):
    axes[0].axvline(e * steps_per_epoch, color="gray", linestyle="--", linewidth=0.5)

axes[0].set_xlabel("Step")
axes[0].set_ylabel("Train Loss (log scale)")
axes[0].set_yscale("log")
axes[0].yaxis.set_major_formatter(plt.ScalarFormatter())
axes[0].set_yticks([0.15, 0.5, 1.0, 2.0])
axes[0].set_title("Train Loss (smoothed)")
axes[0].legend()

# Plot 2 — Test accuracy over epochs
for kind, runs in all_results.items():
    accs = np.array([r["test_accs"] for r in runs])
    mean, std = accs.mean(0), accs.std(0)
    epochs_x = range(1, len(mean) + 1)
    axes[1].plot(epochs_x, mean, label=kind, marker="o")
    axes[1].fill_between(epochs_x, mean - std, mean + std, alpha=0.2)

axes[1].set_xlabel("Epoch")
axes[1].set_xticks(range(1, cfg.epochs + 1))
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Test Accuracy over Epochs")
axes[1].legend()

kinds = list(all_results.keys())
means = [np.mean([r["final_acc"] for r in all_results[k]]) for k in kinds]
stds = [np.std([r["final_acc"] for r in all_results[k]]) for k in kinds]
best_means = [np.mean([r["best_acc"] for r in all_results[k]]) for k in kinds]

color_map = {
    kind: line.get_color()
    for kind, line in zip(all_results.keys(), axes[1].get_lines())
}

for kind, mean, std, best in zip(kinds, means, stds, best_means):
    c = color_map[kind]
    axes[2].errorbar(
        kind,
        mean,
        yerr=std,
        fmt="o",
        capsize=5,
        markersize=5,
        color=c,
        ecolor="black",
        capthick=1.5,
        label=kind,
    )
    # axes[2].plot(kind, best, marker='*', markersize=8, color=c, alpha=0.6)

axes[2].set_ylim(0.9, 1.0)
axes[2].set_ylabel("Test Accuracy")
axes[2].set_title("Final Accuracy (mean ± std, 3 seeds)")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("vit_results.png", dpi=150)
plt.show()

print("hello")
import pickle
import numpy as np

with open("vit_results.pkl", "rb") as f:
    all_results = pickle.load(f)

print(f"{'Method':<12} {'Final Acc (mean)':<18} {'Std':<10}")
print("-" * 40)

for method in all_results.keys():
    final_accs = np.array([run["final_acc"] for run in all_results[method]])
    mean = final_accs.mean()
    std = final_accs.std()
    print(f"{method:<12} {mean:.4f} ± {std:.4f}       {std:.4f}")
