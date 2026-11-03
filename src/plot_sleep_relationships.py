"""نمودارهای توصیفی خواب و پنج متغیر توضیحی، همراه با مجموعه هفت‌نموداری."""

from pathlib import Path

from plot_sleep_distributions import (
    FIGURES, REPORTS, fa, draw as draw_distribution, main as plot_distributions,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from audit_adults import markdown_table
from describe_categories import CATEGORIES


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "DMDEDUC2": ("تحصیلات", "weekday_sleep_by_education"),
    "RIAGENDR": ("جنس", "weekday_sleep_by_sex"),
    "DMDMARTZ": ("وضعیت تأهل", "weekday_sleep_by_marital_status"),
}
CONTINUOUS = {
    "INDFMPIR": ("نسبت درآمد خانواده به خط فقر", "weekday_sleep_by_income", 5),
    "RIDAGEYR": ("سن", "weekday_sleep_by_age", 80),
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: fa(f"{v:g}")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: fa(f"{v:g}")))


def footnote(fig):
    fig.text(0.98, 0.045, fa("بزرگسالان 20 ساله و بالاتر؛ بدون وزن نمونه‌گیری و بدون تعدیل سایر متغیرها"),
             ha="right", fontsize=10)
    fig.text(0.98, 0.013, fa("خواب: کد 2 تقریب کمتر از 3 ساعت و کد 14 تقریب 14 ساعت و بیشتر است."),
             ha="right", fontsize=9)


def paired_data(adults, col):
    valid = adults.SLD012.notna() & adults[col].notna()
    if col in GROUPS:
        valid &= adults[col].isin(CATEGORIES[col][1])
    return adults.loc[valid, ["SEQN", col, "SLD012", "WTINT2YR"]].copy()


def make_education_means(adults, col):
    data = paired_data(adults, col)
    labels = CATEGORIES[col][1]
    rows = []
    for code, label in labels.items():
        group = data.loc[data[col].eq(code)]
        if group.empty or not np.isfinite(group.WTINT2YR).all() or not group.WTINT2YR.gt(0).all():
            raise ValueError("گروه تحصیلی یا وزن آن معتبر نیست.")
        rows.append((label, len(group), float(group.SLD012.mean()),
                     float(np.average(group.SLD012, weights=group.WTINT2YR))))
    table = pd.DataFrame(rows, columns=["گروه تحصیلی", "تعداد پاسخ مشترک", "میانگین بدون وزن؛ ساعت", "میانگین وزندار؛ ساعت"])
    table.to_csv(REPORTS / "education_mean_weight_comparison.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 6.8))
    style(ax)
    positions = np.arange(len(rows))
    ax.scatter(table.iloc[:, 2], positions-0.10, marker="o", facecolors="none", edgecolors="#6b747d",
               s=65, linewidths=1.5, label=fa("میانگین بدون وزن"), zorder=3)
    ax.scatter(table.iloc[:, 3], positions+0.10, marker="D", color="#397a91",
               s=45, label=fa("میانگین با وزن مصاحبه"), zorder=3)
    for index, row in table.iterrows():
        ax.annotate(fa(f"{row.iloc[2]:.3f}"), (row.iloc[2],index-0.10), xytext=(10,4),
                    textcoords="offset points", fontsize=9, color="#6b747d")
        ax.annotate(fa(f"{row.iloc[3]:.3f}"), (row.iloc[3],index+0.10), xytext=(10,-7),
                    textcoords="offset points", fontsize=9, color="#397a91")
    ax.set_yticks(positions, [fa(f"{label}\nتعداد: {n}") for label,n,_,_ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0,10)
    ax.set_xticks(np.arange(0,11,2))
    ax.grid(axis="x", color="#e0e5e8", linewidth=0.7)
    ax.set_xlabel(fa("میانگین تقریبی خواب روزهای کاری (ساعت)"), labelpad=10)
    ax.set_title(fa("میانگین خواب و تحصیلات؛ مقایسه بدون وزن و وزندار"), fontsize=15, pad=28)
    ax.text(1,1.025,fa(f"پاسخ موجود برای هر دو متغیر: {len(data)} نفر | اطلاعات ناقص: {len(adults)-len(data)} نفر"),
            ha="right",transform=ax.transAxes,fontsize=10)
    ax.legend(loc="lower left",frameon=False,fontsize=9)
    fig.subplots_adjust(left=0.38,right=0.97,top=0.81,bottom=0.21)
    fig.text(0.98,0.045,fa("افراد دو محاسبه در هر گروه یکسان‌اند؛ بدون تعدیل سایر متغیرها و بدون فاصله اطمینان"),ha="right",fontsize=9)
    fig.text(0.98,0.013,fa("خواب: کد 2 تقریب کمتر از 3 ساعت و کد 14 تقریب 14 ساعت و بیشتر است."),ha="right",fontsize=9)
    return fig, GROUPS[col][1], data


def make_box(adults, col):
    label, stem = GROUPS[col]
    data = paired_data(adults, col)
    labels = CATEGORIES[col][1]
    samples = [data.loc[data[col].eq(code), "SLD012"] for code in labels]
    if any(len(sample) == 0 for sample in samples):
        raise ValueError("یکی از گروه‌های نمودار مشاهده‌ای ندارد.")
    fig, ax = plt.subplots(figsize=(12, 6.8 if col == "DMDEDUC2" else 5.6))
    style(ax)
    ax.boxplot(samples, orientation="horizontal", patch_artist=True, widths=0.52,
               boxprops={"facecolor": "#b6d2da", "edgecolor": "#397a91"},
               medianprops={"color": "#8b4030", "linewidth": 2},
               whiskerprops={"color": "#397a91"}, capprops={"color": "#397a91"},
               flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#397a91",
                           "markeredgewidth": 0, "alpha": 0.28})
    ax.set_yticks(np.arange(1, len(labels) + 1),
                  [fa(f"{name}\nتعداد: {len(sample)}") for name, sample in zip(labels.values(), samples)])
    ax.invert_yaxis()
    ax.set_xlim(1.5, 14.5)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
    ax.grid(axis="x", color="#e0e5e8", linewidth=0.7)
    ax.set_xlabel(fa("مدت خواب کدگذاری‌شده روزهای کاری (ساعت)"), labelpad=10)
    ax.set_title(fa(f"خواب روزهای کاری بر حسب {label}"), fontsize=15, pad=28)
    ax.text(1, 1.025, fa(f"پاسخ موجود برای هر دو متغیر: {len(data)} نفر | اطلاعات ناقص: {len(adults)-len(data)} نفر"),
            ha="right", transform=ax.transAxes, fontsize=10)
    fig.subplots_adjust(left=0.38 if col == "DMDEDUC2" else 0.27,
                        right=0.98, top=0.81, bottom=0.21)
    footnote(fig)
    return fig, stem, data


def make_density(adults, col):
    label, stem, cap = CONTINUOUS[col]
    data = paired_data(adults, col)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    style(ax)
    extent = (0, 5.05, 1.75, 14.25) if col == "INDFMPIR" else (19.5, 80.5, 1.75, 14.25)
    collection = ax.hexbin(data[col], data.SLD012, gridsize=(40, 24), mincnt=1,
                           extent=extent, cmap="Blues", linewidths=0, norm=LogNorm(vmin=1))
    if not np.isclose(collection.get_array().sum(), len(data)):
        raise ValueError("همه مشاهدات وارد نمودار تراکم نشده‌اند.")
    colorbar = fig.colorbar(collection, ax=ax, pad=0.025)
    colorbar.set_label(fa("تعداد در هر خانه؛ مقیاس لگاریتمی رنگ"), labelpad=12)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: fa(f"{v:g}")))
    ax.set_ylim(1.5, 14.5)
    ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
    ax.set_ylabel(fa("مدت خواب کدگذاری‌شده روزهای کاری (ساعت)"))
    ax.set_xlabel(fa(label + (" (سال)" if col == "RIDAGEYR" else " (بدون واحد)")), labelpad=10)
    if col == "INDFMPIR":
        ax.set_xlim(-0.05, 5.1)
        ax.set_xticks([0, 1, 2, 3, 4, 5], [fa(v) for v in [0, 1, 2, 3, 4, "5 و بیشتر"]])
    else:
        ax.set_xlim(19, 81)
        ax.set_xticks([20, 30, 40, 50, 60, 70, 80], [fa(v) for v in [20, 30, 40, 50, 60, 70, "80 و بیشتر"]])
    ax.axvline(cap, linestyle="--", linewidth=1, color="#ae654b", alpha=0.8)
    ax.set_title(fa(f"خواب روزهای کاری و {label}"), fontsize=15, pad=28)
    ax.text(1, 1.025, fa(f"پاسخ موجود برای هر دو متغیر: {len(data)} نفر | اطلاعات ناقص: {len(adults)-len(data)} نفر"),
            ha="right", transform=ax.transAxes, fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.88, top=0.81, bottom=0.22)
    footnote(fig)
    return fig, stem, data


def main():
    # دو نمودار توزیع و تنظیمات قلم با همان کد قبلی بازتولید می‌شوند.
    plot_distributions()
    demo = pd.read_sas(ROOT / "data/raw/DEMO_L.xpt", format="xport", encoding=None)
    sleep = pd.read_sas(ROOT / "data/raw/SLQ_L.xpt", format="xport", encoding=None)
    adults = demo.merge(sleep, on="SEQN", how="left", validate="one_to_one").loc[lambda t: t.RIDAGEYR.ge(20)]
    specifications = [
        ("DMDEDUC2", make_education_means), ("INDFMPIR", make_density),
        ("RIDAGEYR", make_density), ("RIAGENDR", make_box), ("DMDMARTZ", make_box),
    ]
    counts, group_stats, figures = [], [], []
    metadata = {"CreationDate": None, "ModDate": None, "Title": "نمودارهای توصیفی مدت خواب"}
    supporting = FIGURES / "supporting"
    supporting.mkdir(exist_ok=True)
    previous_box, _, _ = make_box(adults, "DMDEDUC2")
    for extension in ("png", "pdf"):
        previous_box.savefig(supporting / f"education_boxplot.{extension}", dpi=180,
                             metadata=metadata if extension == "pdf" else None)
    plt.close(previous_box)
    with PdfPages(FIGURES / "all_seven_plots.pdf", metadata=metadata) as pdf:
        for col in ["SLD012", "SLD013"]:
            fig, ax = plt.subplots(figsize=(12, 5.8))
            draw_distribution(ax, col, adults[col], 25, adults.WTINT2YR if col == "SLD012" else None)
            note = "افراد دو محاسبه یکسان‌اند؛ وزن رسمی مصاحبه | ستون‌های هاشورخورده: گروه‌های انتهایی" if col == "SLD012" else "بزرگسالان 20 ساله و بالاتر؛ بدون وزن نمونه‌گیری | ستون‌های قهوه‌ای: گروه‌های انتهایی"
            fig.text(0.98, 0.025, fa(note), ha="right", fontsize=10)
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.82)
            pdf.savefig(fig)
            plt.close(fig)
        for col, maker in specifications:
            fig, stem, data = maker(adults, col)
            for extension in ("png", "pdf"):
                fig.savefig(FIGURES / f"{stem}.{extension}", dpi=180,
                            metadata=metadata if extension == "pdf" else None)
            pdf.savefig(fig)
            plt.close(fig)
            title = GROUPS[col][0] if col in GROUPS else CONTINUOUS[col][0]
            counts.append((title, len(data), len(adults)-len(data)))
            figures.append((title, stem))
            if col in GROUPS:
                for code, group in data.groupby(col, sort=True):
                    values = group.SLD012
                    inner = values.loc[~values.isin([2, 14])]
                    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
                    group_stats.append((title, CATEGORIES[col][1][code], len(values), float(values.mean()),
                                        median, q1, q3, int(values.isin([2,14]).sum()),
                                        float(inner.mean()) if len(inner) else np.nan))
    count_table = pd.DataFrame(counts, columns=["متغیر توضیحی", "تعداد واردشده در نمودار", "اطلاعات ناقص در دست‌کم یکی از دو متغیر"])
    count_table.to_csv(REPORTS / "relationship_plot_counts.csv", index=False, encoding="utf-8-sig")
    summaries = pd.DataFrame(group_stats, columns=["متغیر", "گروه", "تعداد", "میانگین تقریبی خواب", "میانه", "چارک اول", "چارک سوم", "تعداد دو گروه انتهایی", "میانگین بدون دو گروه انتهایی"])
    summaries.to_csv(REPORTS / "sleep_by_group_summaries.csv", index=False, encoding="utf-8-sig")
    display_summaries = summaries.copy()
    for col in display_summaries.select_dtypes(include="floating"):
        display_summaries[col] = display_summaries[col].map(lambda value: f"{value:.3f}")
    sections = "\n\n".join(f"## خواب و {title}\n\n![خواب و {title}](../../figures/phase1/{stem}.png)" for title, stem in figures)
    report = f"""# خواب روزهای کاری و متغیرهای توضیحی

چون مدت خواب روزهای کاری متغیر پاسخ اصلی مطالعه است، رابطه توصیفی آن را با تحصیلات، نسبت درآمد به خط فقر، سن، جنس و وضعیت تأهل نشان داده‌ایم؛ این مقایسه‌ها دو متغیره‌اند و سایر متغیرها در آن‌ها تعدیل نشده‌اند.

## نمونه هر نمودار

{markdown_table(count_table)}

در هر مقایسه، فقط پاسخ‌های موجود همان دو متغیر وارد نمودار شده‌اند و کدهای امتناع از پاسخ و ندانستن پاسخ در تحصیلات و تأهل، پاسخ موجود محسوب نشده‌اند؛ بنابراین، گمشدگی درآمد باعث حذف فرد از نمودار تحصیلات یا جنس نشده است. در نمودار تحصیلات، میانگین بدون وزن و با وزن مصاحبه روی افراد یکسان هر گروه مقایسه می‌شوند، درحالی‌که چهار نمودار دیگر توصیف بدون وزن نمونه‌اند و هیچ‌کدام برای سایر متغیرها تعدیل نشده‌اند.

## دلیل انتخاب نمودارها

برای تحصیلات از نمودار نقطه‌ای میانگین‌های بدون وزن و وزندار استفاده کرده‌ایم تا نقش وزن‌دهی در مقایسه گروه‌ها دیده شود؛ محور از صفر آغاز می‌شود و نقاط، میانگین تقریبی هر گروه را نشان می‌دهند، نه مقدار خواب افراد یا فاصله اطمینان. برای بررسی میانه و پراکندگی نیز [نمودار جعبه‌ای تحصیلات](../../figures/phase1/supporting/education_boxplot.pdf) در دسترس است.

برای جنس و وضعیت تأهل، نمودار جعبه‌ای را حفظ کرده‌ایم تا مقایسه مرکز و پراکندگی نمونه نیز در مجموعه باقی بماند؛ خط داخل جعبه میانه، دو انتهای آن چارک‌ها و سبیل‌ها دورترین مقادیر در فاصله حداکثر 1.5 برابر دامنه میان‌چارکی از چارک‌ها را نشان می‌دهند. نقاط بیرون سبیل‌ها حذف نشده‌اند، اما چون بعضی نقاط روی هم قرار می‌گیرند، تعداد افراد هر گروه جداگانه نوشته شده است.

برای سن و نسبت درآمد از نمایش تراکم نقاط با خانه‌های شش‌ضلعی استفاده کرده‌ایم، زیرا تعداد زیاد مشاهدات و ثبت خواب در نیم‌ساعت‌ها باعث هم‌پوشانی نقاط می‌شود؛ در این نمایش، تیرگی رنگ تعداد مشاهدات هر خانه را نشان می‌دهد و راهنمای رنگ هر نمودار نیز همین تعداد را مشخص می‌کند. مقیاس رنگ لگاریتمی است تا خانه‌های کم‌جمعیت در کنار خانه‌های پرتراکم قابل مشاهده بمانند، اما محورهای سن، درآمد و خواب خطی‌اند و هیچ مقداری برای تحلیل لگاریتم‌گیری نشده است. این خانه‌بندی صرفاً برای نمایش است و متغیرها را برای تحلیل به گروه‌های تازه تبدیل نمی‌کند؛ همچنین، رنگ‌ها فراوانی مطلق‌اند و احتمال شرطی یا میانگین خواب در هر سن یا سطح درآمد را نشان نمی‌دهند.

در پنج نمودار، کدهای 2 و 14 خواب به‌عنوان تقریب عددی به کار رفته‌اند و این فرض زیر شکل‌ها نوشته شده است، درحالی‌که مقادیر 80 سن و 5 نسبت درآمد با عنوان «و بیشتر» مشخص شده‌اند. چهار نمودار افراد، مقیاس خواب یکسان دارند و مقادیر انتهایی را نیز نشان می‌دهند، اما نمودار تحصیلات محور میانگین گروه‌هاست و نباید دامنه آن را دامنه خواب افراد دانست.

{sections}

## جدول‌های پشتیبان

[خلاصه عددی گروه‌ها](sleep_by_group_summaries.csv) و [تحلیل حساسیت میانگین‌های وزندار](weighted_descriptive_comparisons.md) برای بررسی بیشتر در دسترس‌اند.

## دامنه تفسیر

هم‌پوشانی جعبه‌ها یا تفاوت ارتفاع و تیرگی بخش‌های نمودار، به‌تنهایی آزمون فرض نیست و از آن نمی‌توان درباره معنی‌داری آماری یا رابطه علّی نتیجه گرفت؛ افزون بر این، ممکن است بخشی از تفاوت مشاهده‌شده میان گروه‌های تحصیلی یا اقتصادی با تفاوت ترکیب سنی، جنسی یا تأهل آن‌ها همراه باشد، که بررسی هم‌زمان آن‌ها به مرحله رگرسیون چندگانه مربوط است.

## مجموعه کامل و اجرای دوباره

[مجموعه نمودارها](../../figures/phase1/all_seven_plots.pdf) با دستور زیر بازتولید می‌شود و جزئیات محیط اجرا در راهنمای داده‌ها آمده است:

```sh
.venv/bin/python src/plot_sleep_relationships.py
```
"""
    (REPORTS / "sleep_relationships.md").write_text(report, encoding="utf-8")
    print(count_table.to_string(index=False))
    print(display_summaries.to_string(index=False))
    print("مجموعه هفت‌نموداری و گزارش ذخیره شدند.")


if __name__ == "__main__":
    main()
