"""Weighted and unweighted empirical probabilities on identical respondent sets."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(root):
    reports = root / "reports/phase2"
    figures = root / "figures/phase2"
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    demo = pd.read_sas(root / "data/raw/DEMO_L.xpt", format="xport")
    slq = pd.read_sas(root / "data/raw/SLQ_L.xpt", format="xport")
    adults = demo.loc[demo.RIDAGEYR.ge(20)].merge(slq, on="SEQN", how="left", validate="one_to_one")
    sleep = adults.loc[adults.SLD012.notna()]
    if not np.isfinite(sleep.WTINT2YR).all() or not sleep.WTINT2YR.gt(0).all():
        raise ValueError("Invalid interview weights")
    valid = sleep.loc[sleep.DMDEDUC2.isin(range(1, 6))]
    labels = pd.read_csv(root / "reports/phase1/categorical_frequencies.csv")
    labels = labels.loc[labels["نام متغیر"].eq("DMDEDUC2")].set_index("کد")["گروه"]
    samples = [("همه پاسخ‌های معتبر خواب", sleep), ("خواب و تحصیلات معتبر", valid),
               ("پایین‌تر از کارشناسی (چهار سطح)", valid.loc[valid.DMDEDUC2.lt(5)])]
    rows = []
    distribution = []
    for code, group in valid.groupby("DMDEDUC2"):
        samples.append((labels.at[code], group))
        for hour in sorted(sleep.SLD012.unique()):
            selected = group.SLD012.eq(hour)
            distribution.append((int(code), labels.at[code], hour, int(selected.sum()), len(group),
                                 100 * selected.mean(), 100 * np.average(selected, weights=group.WTINT2YR)))
    for label, group in samples:
        short = group.SLD012.lt(7)
        raw = 100 * short.mean()
        weighted = 100 * np.average(short, weights=group.WTINT2YR)
        rows.append((label, len(group), int(short.sum()), raw, weighted, weighted - raw))
    columns = ["گروه", "تعداد", "خواب کمتر از 7 ساعت", "درصد بدون وزن", "درصد وزندار", "اختلاف واحد درصد"]
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(reports / "weighted_short_sleep.csv", index=False, encoding="utf-8-sig")
    dist = pd.DataFrame(distribution, columns=["کد تحصیلات", "گروه", "کد خواب", "فراوانی", *columns[1:2], *columns[3:5]])
    dist.to_csv(reports / "sleep_by_education_distribution.csv", index=False, encoding="utf-8-sig")

    income = sleep.dropna(subset=["INDFMPIR"])
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.scatter(income.INDFMPIR, income.SLD012, s=9, alpha=.12, linewidths=0, color="#156078")
    ax.set(xlabel="نسبت درآمد خانواده به خط فقر", ylabel="خواب روزهای کاری (ساعت)",
           yticks=[2, 4, 6, 8, 10, 12, 14], yticklabels=["کمتر از 3", "4", "6", "8", "10", "12", "14 و بیشتر"],
           xticks=range(6), xticklabels=["0", "1", "2", "3", "4", "5 و بیشتر"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "sleep_income_scatter.png", dpi=170)
    plt.close(fig)
    lines = ["# احتمال خواب کمتر از 7 ساعت با وزن مصاحبه", "",
             "برای هر سطر، یک بار تعداد افراد با خواب کمتر از 7 ساعت را بر تعداد پاسخ‌دهندگان همان سطر تقسیم کردیم. بار دوم، مجموع وزن مصاحبه افراد با این رویداد را بر مجموع وزن همان پاسخ‌دهندگان تقسیم کردیم. ستون آخر تغییر ناشی از وزن‌دهی را بر حسب واحد درصد نشان می‌دهد.", "",
             "| " + " | ".join(columns) + " |", "|---|---:|---:|---:|---:|---:|"]
    for label, n, count, raw, weighted, gap in rows:
        lines.append(f"| {label} | {n} | {count} | {raw:.2f} | {weighted:.2f} | {gap:+.2f} |")
    lines += ["", "سطرهای کل و گروه ترکیبی با سطرهای پنج سطح هم‌پوشانی دارند. برای مثال وزن رسمی مصاحبه، تفاوت احتمال انتخاب و عدم پاسخ در سطح پیمایش را لحاظ می‌کند و سهم افراد را در برآورد تغییر می‌دهد. عدم پاسخ به سؤال تحصیلات یا خواب می‌تواند همچنان بر برآورد اثر بگذارد.",
              "", f"در افراد دارای هر دو پاسخ، اختلاف گروه پایین‌تر از کارشناسی با کارشناسی یا بالاتر از {rows[2][3]-rows[-1][3]:.2f} واحد درصد بدون وزن به {rows[2][4]-rows[-1][4]:.2f} واحد درصد با وزن می‌رسد. این مقایسه پیش از تعدیل سن، جنس و تأهل است."]
    (reports / "sleep_weights.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
