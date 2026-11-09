import matplotlib.pyplot as plt
from tourism.data import load_arrivals
from tourism.paths import FIGURES

arrivals = load_arrivals()

# ---- Chart 1: normal scale ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(arrivals.index, arrivals.values, linewidth=1)
ax.set_title("Monthly tourist arrivals to Sri Lanka, 1971-2025")
ax.set_xlabel("Years")
ax.set_ylabel("Monthly Arrivals")
fig.tight_layout()
fig.savefig(FIGURES / "01_arrivals_linear.png", dpi=150)
plt.close(fig)

# ---- Chart 2: log scale ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(arrivals.index, arrivals.values, linewidth=1)
ax.set_yscale("log")
ax.set_title("Monthly tourist arrivals to Sri Lanka, 1971-2025 (log scale)")
ax.set_xlabel("Year")
ax.set_ylabel("Arrivals (log scale)")
fig.tight_layout()
fig.savefig(FIGURES / "02_arrivals_log.png", dpi=150)
plt.close(fig)

print("Charts saved to:", FIGURES)