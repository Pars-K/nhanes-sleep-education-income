"""نمایش فراوانی نسبی مدت خواب با حفظ عنوان دو گروه انتهایی."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from audit_adults import markdown_table


FIGURES = ROOT / "figures/phase1"
REPORTS = ROOT / "reports/phase1"
CODES = [2, *np.arange(3, 14, 0.5), 14]
VARIABLES = {
    "SLD012": ("خواب روزهای کاری", "weekday_sleep_distribution"),
    "SLD013": ("خواب روزهای غیرکاری", "nonworkday_sleep_distribution"),
}


def fa(text):
    return str(text)


def category_label(code):
    if code == 2:
        return "کمتر از 3"
    if code == 14:
        return "14 و بیشتر"
    return f"{code:g}"


def weighted_percentages(values, weights):
    valid = values.notna()
    selected_weights = weights.loc[valid]
    if not np.isfinite(selected_weights).all() or not selected_weights.gt(0).all():
        raise ValueError("وزن مصاحبه باید موجود و مثبت باشد.")
    totals = selected_weights.groupby(values.loc[valid]).sum().reindex(CODES, fill_value=0)
    result = 100 * totals / selected_weights.sum()
    if not np.isclose(result.sum(), 100):
        raise ValueError("مجموع درصدهای وزندار باید صد باشد.")
    return result


def draw(ax, col, values, ymax, weights=None):
    title, _ = VARIABLES[col]
    counts = values.value_counts().reindex(CODES, fill_value=0)
    percentages = 100 * counts / values.notna().sum()
    colors = ["#ae654b" if code in [2, 14] else "#397a91" for code in CODES]
    x = np.arange(len(CODES))
    if weights is None:
        ax.bar(x, percentages.to_numpy(), width=0.82, color=colors, zorder=3)
    else:
        weighted = weighted_percentages(values, weights)
        ax.bar(x - 0.20, percentages.to_numpy(), width=0.38, color="#b4bdc4",
               label=fa("بدون وزن"), zorder=3)
        ax.bar(x + 0.20, weighted.to_numpy(), width=0.38, color="#397a91",
               label=fa("با وزن مصاحبه"), zorder=3)
        for container in ax.containers[:2]:
            for index in [0, len(CODES)-1]:
                container[index].set_hatch("///")
                container[index].set_edgecolor("#8b4030")
        ax.legend(handles=[Patch(facecolor="#b4bdc4", label=fa("بدون وزن")),
                           Patch(facecolor="#397a91", label=fa("با وزن مصاحبه"))],
                  loc="upper right", frameon=False, fontsize=10)
    ax.set_xticks(x, [fa(category_label(code)) for code in CODES], rotation=50, ha="right")
    ax.tick_params(axis="x", labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: fa(f"{value:g}")))
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e5e8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(fa("مدت خواب یا گروه خواب (ساعت)"), labelpad=10)
    ax.set_ylabel(fa("درصد پاسخ‌های موجود"))
    ax.set_title(fa(title), fontsize=15, pad=28)
    ax.text(1, 1.025, fa(
        f"پاسخ موجود: {int(values.notna().sum())} نفر | اطلاعات ناموجود: {int(values.isna().sum())} نفر"
    ), transform=ax.transAxes, ha="right", fontsize=10)
    return counts, percentages


def main():
    # انتخاب قلم فارسی از قلم‌های نصب‌شده و ثبت نام آن در راهنمای اجرای نمودارها
    font_path = findfont(FontProperties(family="Noto Sans Arabic"), fallback_to_default=False)
    plt.rcParams.update({"font.family": [FontProperties(fname=font_path).get_name(), "DejaVu Sans"],
                         "font.size": 11, "pdf.fonttype": 42, "axes.unicode_minus": False})
    demo = pd.read_sas(ROOT / "data/raw/DEMO_L.xpt", format="xport", encoding=None)
    sleep = pd.read_sas(ROOT / "data/raw/SLQ_L.xpt", format="xport", encoding=None)
    merged = demo.merge(sleep, on="SEQN", how="left", validate="one_to_one")
    adults = merged.loc[merged.RIDAGEYR.ge(20)]
    maxima = []
    for col in VARIABLES:
        values = adults[col]
        if (values.notna() & ~values.isin(CODES)).any():
            raise ValueError(f"مقدار خواب خارج از کدهای مجاز: {col}")
        if values.notna().sum() == 0:
            raise ValueError("برای رسم نمودار پاسخ موجود نداریم.")
        maxima.append(100 * values.value_counts().max() / values.notna().sum())
        if col == "SLD012":
            maxima.append(weighted_percentages(values, adults.WTINT2YR).max())
    ymax = 5 * np.ceil(max(maxima) / 5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    notes = {
        "SLD012": fa("افراد دو محاسبه یکسان‌اند؛ وزن رسمی مصاحبه | ستون‌های هاشورخورده: گروه‌های انتهایی"),
        "SLD013": fa("بزرگسالان 20 ساله و بالاتر؛ بدون وزن نمونه‌گیری | ستون‌های قهوه‌ای: گروه‌های انتهایی"),
    }
    for col, (title, stem) in VARIABLES.items():
        fig, ax = plt.subplots(figsize=(12, 5.8))
        weights = adults.WTINT2YR if col == "SLD012" else None
        counts, percentages = draw(ax, col, adults[col], ymax, weights)
        if int(counts.sum()) != int(adults[col].notna().sum()):
            raise ValueError("تعداد مشاهدات نمودار با پاسخ‌های موجود برابر نیست.")
        if not np.isclose(percentages.sum(), 100):
            raise ValueError("درصدهای نمودار به صد نمی‌رسند.")
        for code, count, percent in zip(CODES, counts, percentages):
            rows.append((col, category_label(code), code, int(count), float(percent)))
        mode = counts.idxmax()
        summaries.append((title, category_label(mode), int(counts.max()),
                          f"{percentages.loc[mode]:.2f}"))
        fig.text(0.98, 0.025, notes[col], ha="right", fontsize=10)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.82)
        for extension in ("png", "pdf"):
            fig.savefig(FIGURES / f"{stem}.{extension}", dpi=180, metadata={"CreationDate": None} if extension == "pdf" else None)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 11.5))
    for ax, col in zip(axes, VARIABLES):
        draw(ax, col, adults[col], ymax, adults.WTINT2YR if col == "SLD012" else None)
    fig.text(0.98, 0.02, fa("روزهای کاری: مقایسه بدون وزن و وزندار | روزهای غیرکاری: توزیع بدون وزن نمونه"), ha="right", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.93, hspace=0.7)
    fig.savefig(FIGURES / "sleep_distributions_comparison.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows, columns=["متغیر", "عنوان مقدار یا گروه", "کد منتشرشده", "فراوانی", "درصد پاسخ‌های موجود"]).to_csv(
        REPORTS / "sleep_distribution_frequencies.csv", index=False, encoding="utf-8-sig")
    weighted = weighted_percentages(adults.SLD012, adults.WTINT2YR)
    pd.DataFrame({
        "کد منتشرشده": CODES,
        "عنوان مقدار یا گروه": [category_label(code) for code in CODES],
        "درصد بدون وزن": 100*adults.SLD012.value_counts().reindex(CODES, fill_value=0).to_numpy()/adults.SLD012.notna().sum(),
        "درصد وزندار": weighted.to_numpy(),
    }).to_csv(REPORTS / "weekday_distribution_weight_comparison.csv", index=False, encoding="utf-8-sig")
    mode_table = pd.DataFrame(summaries, columns=["متغیر", "نما؛ ساعت", "فراوانی نما", "درصد پاسخ‌های موجود"])
    report = f"""# توزیع مدت خواب در روزهای کاری و غیرکاری

برای بررسی شکل توزیع خواب، درصد پاسخ‌های موجود افراد 20 ساله و بالاتر را در دو نمودار با مقیاس عمودی یکسان نشان داده‌ایم؛ در نمودار روزهای کاری، درصدهای بدون وزن و با وزن رسمی مصاحبه کنار هم آمده‌اند تا نقش وزن‌دهی مستقیماً دیده شود، درحالی‌که نمودار روزهای غیرکاری توصیف بدون وزن نمونه را نشان می‌دهد.

## انتخاب نمودار و مقیاس‌ها

چون داده‌های خواب به نزدیک‌ترین نیم‌ساعت گرد شده‌اند و در دو انتهای توزیع نیز گروه‌های باز «کمتر از 3 ساعت» و «14 ساعت و بیشتر» وجود دارند، از نمودار ستونی فراوانی نسبی استفاده کرده‌ایم تا این دو گروه با عنوان واقعی خود باقی بمانند. محور افقی شامل مقادیر و گروه‌های مرتب‌شده است و پهنای یکسان ستون‌ها به معنی پهنای یکسان بازه‌های خواب نیست؛ در نتیجه، این شکل هیستوگرام با بازه‌های پیوسته و هم‌عرض نیست.

در نمودار روزهای کاری، ستون خاکستری درصد بدون وزن و ستون آبی درصد با وزن `WTINT2YR` را نشان می‌دهد و دو گروه انتهایی هاشور خورده‌اند؛ در نمودار روزهای غیرکاری نیز دو گروه انتهایی قهوه‌ای هستند. مخرج درصد بدون وزن تعداد پاسخ‌های موجود و مخرج درصد وزندار مجموع وزن همان پاسخ‌دهندگان است، بنابراین افراد دو محاسبه یکسان‌اند و برای رسم شکل نیازی به تفسیر کدهای 2 و 14 به‌عنوان زمان دقیق نداریم. محور عمودی هر دو نمودار از صفر شروع می‌شود و حد بالای یکسانی دارد تا تغییر مقیاس بر مقایسه اثر نگذارد.

## خواب روزهای کاری

![توزیع خواب روزهای کاری](../../figures/phase1/weekday_sleep_distribution.png)

## خواب روزهای غیرکاری

![توزیع خواب روزهای غیرکاری](../../figures/phase1/nonworkday_sleep_distribution.png)

## برداشت توصیفی

{markdown_table(mode_table)}

جدول بالا خلاصه بدون وزن دو متغیر را نشان می‌دهد و در هر دو، 8 ساعت نماست؛ همچنین، ستون‌های روزهای کاری امکان بررسی تغییر سهم هر مقدار پس از وزن‌دهی را فراهم می‌کنند، بدون آنکه شکل ظاهری نمودار را آزمون فرض بدانیم. برای مقایسه نوع روزها باید ستون‌های بدون وزن را با هم مقایسه کرد، زیرا نمودار روزهای غیرکاری در این مجموعه وزندار نیست.

این دو نمودار توزیع‌های حاشیه‌ای را نشان می‌دهند و چون مجموعه پاسخ‌دهندگان دو متغیر کاملاً یکسان نیست، تفاوت آن‌ها به معنی محاسبه تغییر خواب هر فرد نیست؛ برای چنین مقایسه‌ای باید تنها افراد دارای هر دو پاسخ را در نظر بگیریم و اختلاف خواب را برای هر فرد محاسبه کنیم. همچنین، شکل نمودارها به‌تنهایی برای تأیید یک توزیع نرمال یا نتیجه‌گیری درباره معنی‌داری تفاوت‌ها کافی نیست و در این مرحله آزمون فرض انجام نشده است.

## فایل‌ها و اجرای دوباره

جدول [فراوانی پاسخ‌ها](sleep_distribution_frequencies.csv) و [درصدهای بدون وزن و وزندار روزهای کاری](weekday_distribution_weight_comparison.csv) مبنای نمودارها هستند؛ جزئیات محیط اجرا در راهنمای داده‌ها آمده است.

```sh
.venv/bin/python src/plot_sleep_distributions.py
```
"""
    (REPORTS / "sleep_distributions.md").write_text(report, encoding="utf-8")
    print(mode_table.to_string(index=False))
    print("دو نمودار مستقل، نسخه مقایسه‌ای و جدول فراوانی ذخیره شدند.")


if __name__ == "__main__":
    main()
