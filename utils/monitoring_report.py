import io
import glob
import html
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.model_monitoring as mm

COLORS = {"champion": "#1f6fb2", "control": "#7f7f7f"}
WATCH_COLOR, ALERT_COLOR = "#e08a1e", "#c0392b"
STATUS_COLORS = {"ok": "#2e8b57", "watch": WATCH_COLOR, "alert": ALERT_COLOR}


def load_partitions(monitoring_directory, prefix):
    files = sorted(glob.glob(monitoring_directory + prefix + "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def to_png_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def new_axes(title, ylabel, height=3.4):
    fig, ax = plt.subplots(figsize=(7.6, height))
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Application month")
    ax.grid(alpha=0.25)
    return fig, ax


def no_data(ax, message="No data yet"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#888888")


def month_axis(ax, months):
    ax.set_xticks(months)
    ax.set_xticklabels([m.strftime("%Y-%m") for m in months], rotation=45, ha="right", fontsize=8)


def chart_auc(perf):
    fig, ax = new_axes("Live AUC by application cohort (champion vs frozen control)", "AUC")
    if perf.empty:
        no_data(ax)
    else:
        for arm in ["champion", "control"]:
            g = perf[perf["arm"] == arm].sort_values("cohort")
            ax.plot(g["cohort"], g["auc"], marker="o", color=COLORS[arm], label=f"{arm} (monthly cohort)")
        g = perf[(perf["arm"] == "champion") & perf["auc_rolling3"].notna()].sort_values("cohort")
        if len(g):
            ax.plot(g["cohort"], g["auc_rolling3"], ls="--", marker="s", mfc="white", color=COLORS["champion"], label="champion rolling 3-month")
            ax.plot(g["cohort"], g["auc_alert_level"], ls=":", color=ALERT_COLOR, label="alert level (validation AUC - 0.05)")
        month_axis(ax, sorted(perf["cohort"].unique()))
        ax.legend(fontsize=8, loc="lower left")
    return to_png_base64(fig)


def chart_ks(perf):
    fig, ax = new_axes("KS by application cohort (separation of defaulters and non-defaulters)", "KS")
    if perf.empty:
        no_data(ax)
    else:
        for arm in ["champion", "control"]:
            g = perf[perf["arm"] == arm].sort_values("cohort")
            ax.plot(g["cohort"], g["ks"], marker="o", color=COLORS[arm], label=arm)
        month_axis(ax, sorted(perf["cohort"].unique()))
        ax.legend(fontsize=8)
    return to_png_base64(fig)


def chart_rates(perf):
    fig, ax = new_axes("Actual default rate vs share flagged high risk (champion)", "Share of loans")
    g = perf[perf["arm"] == "champion"].sort_values("cohort") if not perf.empty else perf
    if g.empty:
        no_data(ax)
    else:
        ax.plot(g["cohort"], g["default_rate"], marker="o", color="#333333", label="actual default rate")
        ax.plot(g["cohort"], g["flagged_rate"], marker="o", color=COLORS["champion"], label="flagged high risk")
        ax.plot(g["cohort"], g["precision"], marker=".", ls="--", color="#2e8b57", label="precision of flagged")
        month_axis(ax, sorted(g["cohort"].unique()))
        ax.legend(fontsize=8)
    return to_png_base64(fig)


def chart_psi(drift_sum):
    fig, ax = new_axes("Drift: PSI by application cohort (live month vs training window)", "PSI", 3.6)
    if drift_sum.empty:
        no_data(ax)
    else:
        for arm in ["champion", "control"]:
            g = drift_sum[drift_sum["arm"] == arm].sort_values("cohort")
            ax.plot(g["cohort"], g["score_psi"], marker="o", color=COLORS[arm], label=f"score PSI ({arm})")
        g = drift_sum[(drift_sum["arm"] == "champion") & drift_sum["max_feature_psi"].notna()].sort_values("cohort")
        ax.plot(g["cohort"], g["max_feature_psi"], marker="^", ls="--", color="#8e44ad", label="worst feature PSI (champion)")
        ax.axhline(mm.PSI_WATCH, color=WATCH_COLOR, ls="--", lw=1)
        ax.axhline(mm.PSI_ALERT, color=ALERT_COLOR, ls="--", lw=1)
        ax.text(ax.get_xlim()[1], mm.PSI_WATCH, " watch 0.10", color=WATCH_COLOR, va="bottom", ha="right", fontsize=8)
        ax.text(ax.get_xlim()[1], mm.PSI_ALERT, " alert 0.25", color=ALERT_COLOR, va="bottom", ha="right", fontsize=8)
        ax.set_ylim(0, max(0.3, float(drift_sum["max_feature_psi"].max(skipna=True) or 0) * 1.1))
        month_axis(ax, sorted(drift_sum["cohort"].unique()))
        ax.legend(fontsize=8, loc="upper left")
    return to_png_base64(fig)


def chart_heatmap(drift_detail, top_n=15):
    feats = drift_detail[(drift_detail["item_type"] == "feature")] if not drift_detail.empty else drift_detail
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_title("Feature PSI by application cohort (champion; the 15 features with the highest PSI)", fontsize=11, loc="left")
    if feats.empty:
        no_data(ax)
        return to_png_base64(fig)
    pivot = feats.pivot_table(index="item", columns="cohort", values="psi")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index[:top_n]]
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=mm.PSI_ALERT)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([c[:7] for c in pivot.columns], rotation=45, ha="right", fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="PSI (colour scale tops out at the 0.25 alert level)")
    return to_png_base64(fig)


def chart_confusion(perf):
    # pooled over every matured cohort so far, at the threshold saved with each model
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    fig.suptitle("Confusion matrix at the saved threshold (all matured cohorts pooled)", fontsize=11, x=0.01, ha="left")
    for ax, arm in zip(axes, ["champion", "control"]):
        g = perf[perf["arm"] == arm] if not perf.empty else perf
        ax.set_title(arm, fontsize=10)
        if g.empty or g["tp"].isna().all():
            no_data(ax)
            ax.axis("off")
            continue
        tp, fp, fn, tn = (int(g[c].sum()) for c in ["tp", "fp", "fn", "tn"])
        counts = np.array([[tn, fp], [fn, tp]])
        row_share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{counts[i, j]:,}\n{row_share[i, j]:.0%} of row", ha="center", va="center", fontsize=9,
                        color="white" if row_share[i, j] > 0.6 else "#222222")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["not flagged", "flagged"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["did not default", "defaulted"], fontsize=8)
        ax.set_xlabel(f"precision {tp / (tp + fp):.0%}  |  recall {tp / (tp + fn):.0%}  |  false alarms {fp / (fp + tn):.0%}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return to_png_base64(fig)


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "&ndash;"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return html.escape(str(v))


def table_html(df, cols, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in cols:
            v = r[c]
            if c == "status":
                cells += f"<td><span class='pill' style='background:{STATUS_COLORS.get(v, '#888')}'>{html.escape(str(v))}</span></td>"
            elif c in ("n_loans",) and not pd.isna(v):
                cells += f"<td>{int(v)}</td>"
            else:
                cells += f"<td>{fmt(v)}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;background:#fff}
h1{font-size:24px;margin-bottom:2px}h2{font-size:17px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:4px}
.sub{color:#666;font-size:13px;margin-bottom:16px}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.card{border:1px solid #ddd;border-radius:8px;padding:10px 14px;min-width:150px}.card b{display:block;font-size:20px}
.pill{color:#fff;border-radius:10px;padding:1px 9px;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #ddd;padding:4px 6px;text-align:right}
th{background:#f3f3f3}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
img{max-width:100%}.note{font-size:13px;color:#444;background:#f7f7f7;border-left:3px solid #1f6fb2;padding:8px 12px;margin:8px 0}
"""


def build_report(monitoring_directory, output_path):
    summary = load_partitions(monitoring_directory, "gold_monitoring_summary_")
    detail = load_partitions(monitoring_directory, "gold_monitoring_drift_")
    if summary.empty:
        print("no monitoring partitions found in", monitoring_directory)
        return None
    summary["cohort"] = pd.to_datetime(summary["perf_cohort"])
    perf = summary[summary["auc"].notna()].copy()
    drift_sum = summary[summary["score_psi"].notna()].copy()
    drift_sum["cohort"] = pd.to_datetime(drift_sum["drift_cohort"])
    if not detail.empty:
        detail["cohort"] = detail["cohort"].astype(str)

    latest = summary["run_month"].max()
    last = summary[(summary["run_month"] == latest) & (summary["arm"] == "champion")].iloc[0]
    n_runs = summary["run_month"].nunique()
    n_alert = int(((summary["arm"] == "champion") & (summary["status"] == "alert")).sum())
    n_watch = int(((summary["arm"] == "champion") & (summary["status"] == "watch")).sum())
    n_trig = int(((summary["arm"] == "champion") & summary["retrain_trigger"]).sum())

    cards = (f"<div class='card'>Latest run<b>{latest[:7]}</b></div>"
             f"<div class='card'>Champion status<b style='color:{STATUS_COLORS[last['status']]}'>{last['status'].upper()}</b></div>"
             f"<div class='card'>Runs monitored<b>{n_runs}</b></div>"
             f"<div class='card'>Watch / alert runs<b>{n_watch} / {n_alert}</b></div>"
             f"<div class='card'>Retrain triggers<b>{n_trig}</b></div>")

    cols = ["run_month", "arm", "status", "drift_cohort", "score_psi", "max_feature_psi", "perf_cohort", "n_loans", "auc", "ks",
            "auc_rolling3", "auc_alert_level", "retrain_trigger"]
    heads = ["Run month", "Arm", "Status", "Scored cohort", "Score PSI", "Worst feature PSI", "Matured cohort", "Loans", "AUC", "KS",
             "Rolling 3m AUC", "Alert level", "Retrain trigger"]
    table = table_html(summary.sort_values(["run_month", "arm"]), cols, heads)

    sections = [
        ("Performance (needs outcomes, so it appears 6 months after scoring)", [
            (chart_auc(perf), "Each point is a monthly cohort scored when the loans were applied for and evaluated once outcomes arrived. The champion line is the managed model, the control is the frozen first model. The red dotted line is the alert level: validation AUC of the model minus 0.05. A retrain trigger needs the rolling AUC below it two months in a row."),
            (chart_ks(perf), "KS is the largest gap between the cumulative score curves of defaulters and non-defaulters. Higher means the score separates the two groups better."),
            (chart_rates(perf), "Share of loans flagged high risk against the share that actually defaulted, and the precision of the flags."),
            (chart_confusion(perf), "Rows are what really happened, columns are what the model said. Bottom-right: defaulters correctly flagged. Top-right: good customers flagged by mistake (false alarms). Bottom-left: defaulters the model missed. Precision is the share of flagged loans that really default; recall is the share of defaulters that get flagged; false alarms is the share of good customers that get flagged.")]),
        ("Drift (no outcomes needed, available in the month of scoring)", [
            (chart_psi(drift_sum), "PSI measures stability, not accuracy. On the features it asks whether this month's applicants look like the customers the model was trained on (input stability); on the score it asks whether the model's output has shifted (output stability). Below 0.10 is stable, 0.10 to 0.25 is watch, above 0.25 is alert. With about 500 loans a month, PSI of 0.02 to 0.05 is normal noise."),
            (chart_heatmap(detail), "Feature-level PSI for the champion: how stable each input is compared with the training window. Darker means the applicants changed more on that feature; the colour scale reaches full red at the 0.25 alert level.")]),
    ]
    body = ""
    for title, items in sections:
        body += f"<h2>{title}</h2>"
        for img, caption in items:
            body += f"<img src='data:image/png;base64,{img}'><div class='note'>{caption}</div>"

    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Model monitoring report</title><style>{CSS}</style></head><body>"
            f"<h1>Model monitoring report</h1><div class='sub'>Credit default model, run months up to {latest[:7]}. "
            f"Triggers: PSI alert at 0.25, or live rolling 3-month AUC more than 0.05 below validation AUC for 2 months in a row.</div>"
            f"<div class='cards'>{cards}</div>{body}<h2>Monitoring table</h2>{table}</body></html>")
    with open(output_path, "w") as f:
        f.write(page)
    print("report saved to:", output_path)
    return output_path