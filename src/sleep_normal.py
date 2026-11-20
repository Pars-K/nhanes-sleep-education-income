"""Compare recorded weekday sleep with a normal reference, using phase 1 frequencies."""

from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(root, output_root):
    phase1 = root / "reports/phase1"
    frequencies = pd.read_csv(phase1 / "sleep_distribution_frequencies.csv")
    sleep = frequencies.loc[frequencies["متغیر"].eq("SLD012")].sort_values("کد منتشرشده")
    codes = sleep["کد منتشرشده"].to_numpy()
    counts = sleep["فراوانی"].to_numpy()
    values = np.repeat(codes, counts)
    n = len(values)
    mean, sd = values.mean(), values.std(ddof=1)
    summary = pd.read_csv(phase1 / "numeric_summaries.csv").set_index("نام متغیر").loc["SLD012"]
    if n != summary["تعداد"] or not np.allclose([mean, sd], summary[["میانگین", "انحراف معیار"]].to_numpy(float)):
        raise ValueError("Sleep counts or summary statistics disagree with phase 1.")
    model = NormalDist(mean, sd)
    # End categories are open. Draw only the finite half-hour bins, using all responses as denominator.
    interior = (codes >= 3) & (codes < 14)
    x = codes[interior]
    observed = 100 * counts[interior] / n
    expected = np.array([100 * (model.cdf(t + .25) - model.cdf(t - .25)) for t in x])
    empirical_cdf = 100 * counts.cumsum()[interior] / n
    model_cdf = [100 * model.cdf(t + .25) for t in x]
    figures = output_root / "figures/phase2"
    reports = output_root / "reports/phase2"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    ax, cumulative = axes
    ax.bar(x, observed, width=.5, color="#277589", edgecolor="white", label="مشاهده‌شده")
    ax.plot(x, expected, "o-", color="#a34824", markersize=3, label="تقریب نرمال")
    ax.set(title="فراوانی در رده‌های نیم‌ساعته", ylabel="درصد کل پاسخ‌ها")
    cumulative.plot(x, empirical_cdf, "o", ms=4, color="#277589", label="مشاهده‌شده")
    cumulative.plot(x, model_cdf, color="#a34824", label="تقریب نرمال")
    cumulative.set(title="سهم پاسخ‌ها تا هر رده", ylabel="درصد تجمعی", ylim=(0, 103))
    for panel in axes:
        panel.set(xlabel="خواب ثبت‌شده در روز کاری (ساعت)", xticks=range(3, 14))
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", alpha=.2)
        panel.set_axisbelow(True)
        panel.legend(frameon=False)
    fig.savefig(figures / "sleep_normal.png", dpi=160)
    plt.close(fig)

    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    skew = np.mean(((values - mean) / values.std()) ** 3)
    at_eight = x == 8
    text = f"""# شکل توزیع خواب روزهای کاری و تقریب نرمال

این بخش بر پایه [فراوانی‌های فاز اول](../phase1/sleep_distribution_frequencies.csv) و همه {n} پاسخ معتبر بزرگسالان برای `SLD012` تهیه شد. اینجا توزیع خود مدت خواب را بدون وزن پیمایش بررسی می‌کنیم. میانگین {mean:.2f} و انحراف معیار نمونه {sd:.2f} ساعت است. منحنی مرجع نرمال با همین دو پارامتر رسم شد. مطابق فاز اول، برای محاسبه شاخص‌های عددی، کدهای 2 و 14 تقریب دو رده انتهایی هستند.

![هیستوگرام رده‌های نیم‌ساعته و مقایسه توزیع تجمعی با مدل نرمال](../../figures/phase2/sleep_normal.png)

در نمودار چپ هر ستون سهم یک رده نیم‌ساعته از کل پاسخ‌هاست. رده‌های «کمتر از 3 ساعت» ({counts[0]} نفر) و «14 ساعت و بیشتر» ({counts[-1]} نفر) پهنای مشخصی ندارند و جداگانه گزارش شده‌اند. برای مقایسه روی مقیاس یکسان، احتمال مدل نرمال را در بازه نیم‌ساعته پیرامون هر مقدار حساب کردیم. در نمودار راست، هر نقطه سهم تجمعی تا همان رده، با احتساب گروه پایین، و خط، احتمال نرمال تا لبه بالایی آن رده است.

میانه {median:g} ساعت و نیمه میانی پاسخ‌ها بین {q1:g} و {q3:g} ساعت است. ضریب چولگی گشتاوری مقادیر کدگذاری‌شده {skew:.4f} است. با معیار توصیفی 1.5 برابر دامنه میان‌چارکی، {sum(values < lower)} پاسخ زیر {lower:g} ساعت و {sum(values > upper)} پاسخ بالای {upper:g} ساعت قرار می‌گیرند. این پاسخ‌ها در تحلیل حفظ شده‌اند و طول دقیق خواب در دو رده باز قابل بازیابی نیست.

سهم رده 8 ساعت در داده {observed[at_eight].item():.2f} درصد است. مدل نرمال برای بازه 7.75 تا 8.25 ساعت، {expected[at_eight].item():.2f} درصد پیش‌بینی می‌کند. هر دو توزیع در حوالی 8 ساعت متمرکزند، اما فراوانی برخی ساعت‌های کامل از آنچه منحنی هموار نشان می‌دهد بیشتر است. تقریب نرمال برای توصیف کلی مرکز و پراکندگی مفید است. برای احتمال یک رده خواب مشخص، جدول تجربی الگوی ثبت پاسخ‌ها را بهتر حفظ می‌کند.

برای بازتولید این بخش، از ریشه مخزن `python src/sleep_normal.py` را اجرا می‌کنیم.
"""
    (reports / "sleep_normal.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build(root, root)
