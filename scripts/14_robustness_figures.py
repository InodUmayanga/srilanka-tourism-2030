# =====================================================================
# Figures for the robustness checks (reads saved results, runs no models)
#   figures/14_check_2026.png       forecast vs actual, January-October 2026
#   figures/15_robustness_2030.png  2030 forecast with and without the March-April 2026 shock months
# Run scripts 10, 12 and 13 first.
# =====================================================================
import pandas as pd
import matplotlib.pyplot as plt
from tourism.data import load_arrivals
from tourism.paths import FIGURES, RESULTS

# ---- Figure 14: how the December 2025 forecast did, month by month in 2026 ----
monthly = pd.read_csv(RESULTS / "forecast_vs_actual_2026.csv", index_col=0, parse_dates=True)
check = pd.read_csv(RESULTS / "check_sep_oct_2026.csv")
check = check[check["model"] == "Ensemble (final model)"]
oct_main = check[(check["month"] == "Oct 2026") & (check["trained"] == "trained to Sep 2026")]["forecast"].iloc[0]
oct_robust = pd.read_csv(RESULTS / "robustness_monthly_forecast.csv", index_col=0, parse_dates=True)["ensemble"].iloc[0]
error_pct = (monthly["actual"] - monthly["forecast"]) / monthly["actual"] * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for ax in (ax1, ax2):                                         # shade the Middle East war months
    ax.axvspan(pd.Timestamp("2026-02-15"), pd.Timestamp("2026-04-15"), color="grey", alpha=0.15)
ax1.plot(monthly.index, monthly["actual"], marker="o", linewidth=2, color="black", label="Actual")
ax1.plot(monthly.index, monthly["forecast"], marker="o", linestyle="--", label="Ensemble forecast made in Dec 2025")
october = pd.Timestamp("2026-10-01")
ax1.scatter([october + pd.Timedelta(days=9)], [oct_main], marker="s", s=60, color="tab:green", zorder=3, label="October forecast from Sep 2026 (main)")
ax1.scatter([october + pd.Timedelta(days=18)], [oct_robust], marker="^", s=70, color="tab:orange", zorder=3, label="October forecast from Sep 2026 (shock months disrupted)")
ax1.text(pd.Timestamp("2026-03-15"), ax1.get_ylim()[1] * 0.97, "Middle East war\n(Mar-Apr)", ha="center", va="top", fontsize=9)
ax1.axvline(pd.Timestamp("2026-09-20"), color="grey", linestyle=":")
ax1.text(pd.Timestamp("2026-09-24"), ax1.get_ylim()[0] * 1.03, "published after\nthe sealed test", fontsize=8, va="bottom")
ax1.set_title("2026: forecast made in December 2025 vs actual arrivals")
ax1.set_ylabel("Monthly arrivals")
ax1.legend(fontsize=8, loc="upper right")

colors = ["tab:red" if abs(e) > 10 else "tab:blue" for e in error_pct]
ax2.bar(monthly.index, error_pct, width=20, color=colors)
ax2.axhline(0, color="black", linewidth=0.8)
for d, e in error_pct.items():
    ax2.text(d, e + (1.5 if e >= 0 else -1.5), f"{e:+.1f}%", ha="center", va="bottom" if e >= 0 else "top", fontsize=8)
mape = error_pct.abs().mean()
ax2.set_title(f"Forecast error by month (positive = forecast too low); MAPE Jan-Oct = {mape:.1f}%; red = more than 10% off", fontsize=10)
ax2.set_ylabel("Error (%)")
ax2.set_ylim(error_pct.min() - 8, error_pct.max() + 8)
fig.tight_layout()
fig.savefig(FIGURES / "14_check_2026.png", dpi=150)
plt.close(fig)

# ---- Figure 15: 2030 forecast with and without the shock months ----
main = pd.read_csv(RESULTS / "forecast_2030.csv", index_col=0)
robust = pd.read_csv(RESULTS / "robustness_shock_months.csv", index_col=0)
years = list(main.index)
actual_2025 = load_arrivals()["2025"].sum() / 1e6

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.scatter([2025], [actual_2025], color="black", zorder=3, label="Actual 2025")
offset = 0.12                                                 # shift the two versions apart so both ranges show
ax.errorbar([y - offset for y in years], main["forecast"] / 1e6,
            yerr=[(main["forecast"] - main["low (10%)"]) / 1e6, (main["high (90%)"] - main["forecast"]) / 1e6],
            fmt="o-", capsize=4, label="Main forecast (shock months kept)")
ax.errorbar([y + offset for y in years], robust["shock months disrupted (M)"],
            yerr=[robust["shock months disrupted (M)"] - robust["low 10% (M)"],
                  robust["high 90% (M)"] - robust["shock months disrupted (M)"]],
            fmt="s--", capsize=4, label="March-April 2026 treated as disrupted")
ax.axhline(4, color="orange", linestyle=":", label="4 million target")
ax.axhline(5, color="red", linestyle="--", label="5 million target")
last = years[-1]
ax.annotate(f"{main.loc[last, 'forecast'] / 1e6:.2f}M", (last - offset, main.loc[last, "forecast"] / 1e6),
            xytext=(-14, -22), textcoords="offset points", fontsize=9)
ax.annotate(f"{robust.loc[last, 'shock months disrupted (M)']:.2f}M", (last + offset, robust.loc[last, "shock months disrupted (M)"]),
            xytext=(8, -4), textcoords="offset points", fontsize=9)
ax.set_xticks([2025] + years)
ax.set_title("Robustness: excluding the March-April 2026 shock raises 2030 by "
             f"{robust.loc[last, 'difference %']:.0f}%; both versions stay far below 4 million")
ax.set_ylabel("Annual arrivals (millions), with 80% range")
ax.set_ylim(0, 5.5)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "15_robustness_2030.png", dpi=150)
plt.close(fig)
print("Saved figures/14_check_2026.png and figures/15_robustness_2030.png")
