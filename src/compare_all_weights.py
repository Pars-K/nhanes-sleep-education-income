"""مقایسه توصیفی بدون وزن و وزندار همه متغیرهای اصلی مطالعه."""

from pathlib import Path

import numpy as np
import pandas as pd

from audit_adults import markdown_table
from describe_categories import CATEGORIES
from describe_numeric import VARIABLES, formatted


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase1"


def means(data, column):
    if data.empty:
        raise ValueError(f"مشاهده‌ای برای {column} وجود ندارد.")
    values = data[column].to_numpy()
    weights = data.WTINT2YR.to_numpy()
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("مقادیر و وزن‌ها باید موجود و معتبر باشند.")
    return float(values.mean()), float(np.average(values, weights=weights))


def main():
    demo = pd.read_sas(ROOT / "data/raw/DEMO_L.xpt", format="xport", encoding=None)
    sleep = pd.read_sas(ROOT / "data/raw/SLQ_L.xpt", format="xport", encoding=None)
    if demo.SEQN.isna().any() or sleep.SEQN.isna().any():
        raise ValueError("شناسه گمشده وجود دارد.")
    adults = demo.merge(sleep, on="SEQN", how="left", validate="one_to_one").loc[lambda t: t.RIDAGEYR.ge(20)]
    if not np.isfinite(adults.WTINT2YR).all() or not adults.WTINT2YR.gt(0).all():
        raise ValueError("وزن مصاحبه همه افراد باید موجود و مثبت باشد.")
    category_rows, numeric_rows, group_rows, sensitivity_rows, missing_rows = [], [], [], [], []
    total_weight = adults.WTINT2YR.sum()
    for col, (title, labels, special) in CATEGORIES.items():
        if (adults[col].notna() & ~adults[col].isin([*labels, *special])).any():
            raise ValueError(f"کد تعریف‌نشده در {col}")
        valid = adults[col].isin(labels)
        data = adults.loc[valid]
        denom = data.WTINT2YR.sum()
        missing_rows.append((title, int((~valid).sum()), 100 * (~valid).mean(),
                             100 * adults.loc[~valid, "WTINT2YR"].sum() / total_weight))
        for code, label in labels.items():
            subgroup = data.loc[data[col].eq(code)]
            simple = 100 * len(subgroup) / len(data)
            weighted = 100 * subgroup.WTINT2YR.sum() / denom
            category_rows.append((col, title, label, len(subgroup), len(data), simple, weighted, weighted-simple))
            with_sleep = subgroup.loc[subgroup.SLD012.notna()]
            simple_mean, weighted_mean = means(with_sleep, "SLD012")
            inner = with_sleep.loc[~with_sleep.SLD012.isin([2, 14])]
            inner_simple, inner_weighted = means(inner, "SLD012")
            group_rows.append((col, title, label, len(with_sleep),
                               simple_mean, weighted_mean, 60 * (weighted_mean-simple_mean)))
            sensitivity_rows.append((col, title, label, len(with_sleep)-len(inner), len(inner),
                                     inner_simple, inner_weighted, 60*(inner_simple-simple_mean),
                                     60*(inner_weighted-weighted_mean)))
    for col, (label, unit) in VARIABLES.items():
        valid = adults[col].notna()
        data = adults.loc[valid]
        simple, weighted = means(data, col)
        numeric_rows.append((col, label, unit, len(data), simple, weighted, weighted-simple))
        missing_rows.append((label, int((~valid).sum()), 100*(~valid).mean(),
                             100*adults.loc[~valid, "WTINT2YR"].sum()/total_weight))
        if col in ("SLD012", "SLD013"):
            inner = data.loc[~data[col].isin([2,14])]
            s, w = means(inner, col)
            sensitivity_rows.append((col, label, "کل پاسخ‌های موجود", len(data)-len(inner), len(inner),
                                     s, w, 60*(s-simple), 60*(w-weighted)))
    categories = pd.DataFrame(category_rows, columns=[
        "نام متغیر", "متغیر", "گروه", "فراوانی نمونه", "مخرج تعداد پاسخ‌های موجود",
        "درصد بدون وزن", "درصد وزندار", "تغییر؛ واحد درصد",
    ])
    numeric = pd.DataFrame(numeric_rows, columns=[
        "نام متغیر", "متغیر", "واحد", "تعداد پاسخ موجود", "میانگین بدون وزن", "میانگین وزندار", "تفاوت در واحد متغیر",
    ])
    groups = pd.DataFrame(group_rows, columns=[
        "نام متغیر", "متغیر گروه‌بندی", "گروه", "تعداد پاسخ مشترک", "میانگین خواب بدون وزن؛ ساعت",
        "میانگین خواب وزندار؛ ساعت", "تفاوت وزن‌دهی؛ دقیقه",
    ])
    sensitivity = pd.DataFrame(sensitivity_rows, columns=[
        "نام متغیر", "متغیر یا گروه‌بندی", "گروه", "تعداد دو گروه انتهایی", "تعداد پس از کنارگذاشتن دو گروه",
        "میانگین بدون وزن پس از کنارگذاشتن؛ ساعت", "میانگین وزندار پس از کنارگذاشتن؛ ساعت",
        "تغییر میانگین بدون وزن؛ دقیقه", "تغییر میانگین وزندار؛ دقیقه",
    ])
    missing = pd.DataFrame(missing_rows, columns=[
        "متغیر", "تعداد اطلاعات ناموجود", "درصد ناموجود از تعداد بزرگسالان", "درصد ناموجود از مجموع وزن بزرگسالان",
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"weighted_categorical_comparisons": categories, "weighted_numeric_comparisons": numeric,
              "weighted_sleep_by_group": groups, "weighted_group_sleep_sensitivity": sensitivity,
              "weighted_missingness": missing}
    for filename, table in tables.items():
        table.to_csv(OUT / f"{filename}.csv", index=False, encoding="utf-8-sig")
    for _, table in categories.groupby("نام متغیر"):
        if not np.allclose(table[["درصد بدون وزن", "درصد وزندار"]].sum(), 100):
            raise ValueError("مجموع درصد گروه‌ها باید صد باشد.")
    married = categories.loc[categories["نام متغیر"].eq("DMDMARTZ")].iloc[0]
    age = numeric.loc[numeric["نام متغیر"].eq("RIDAGEYR")].iloc[0]
    never_married = groups.loc[groups["نام متغیر"].eq("DMDMARTZ")].iloc[2]
    report = f"""# جدول‌ها و توضیحات پشتیبان وزن‌دهی

این جدول‌ها درصد گروه‌ها و میانگین‌های بدون وزن و وزندار را نشان می‌دهند، زیرا تغییر اندک میانگین کلی خواب لزوماً به معنی تغییر اندک ترکیب نمونه یا میانگین خواب در هر گروه نیست. سن و نسبت درآمد به‌صورت کمی بررسی شده‌اند و [خلاصه نتایج](weighting_summary.md) نیز جداگانه در دسترس است.

## مبنای وزن‌دهی و مخرج محاسبات

در همه محاسبات از وزن مصاحبه `WTINT2YR` استفاده شده است، زیرا متغیرهای مطالعه از مصاحبه به دست آمده‌اند؛ [روش انتخاب و بررسی وزن](weighted_sleep_comparison.md) توضیح داده شده است و اینجا تنها مخرج هر مقایسه را مشخص می‌کنیم.

درصد وزندار هر گروه از تقسیم مجموع وزن افراد آن گروه بر مجموع وزن پاسخ‌های موجود همان متغیر و ضرب آن در 100 به دست آمده است، درحالی‌که میانگین وزندار از تقسیم مجموع حاصل‌ضرب وزن در مقدار متغیر بر مجموع وزن افراد همان محاسبه به دست می‌آید. برای میانگین خواب درون گروه‌ها نیز وجود هر دو پاسخ لازم است و مخرج فقط به افراد همان گروه با خواب موجود مربوط می‌شود؛ در نتیجه، افراد دو محاسبه بدون وزن و وزندار در هر ردیف یکسان‌اند، اما مخرج جدول ترکیب گروه‌ها ممکن است با مخرج جدول خواب آن‌ها تفاوت داشته باشد.

## ترکیب گروه‌ها

{markdown_table(formatted(categories.drop(columns="نام متغیر")))}

ستون تغییر، درصد وزندار منهای درصد بدون وزن است و واحد آن «واحد درصد» است؛ بنابراین، نباید آن را درصد تغییر نسبی دانست. درصدهای هر متغیر جداگانه به 100 می‌رسند و فراوانی نمونه نیز تعداد افراد واقعی مطالعه است، نه مجموع وزن یا تعداد افراد جامعه.

برای نمونه، سهم گروه متأهل یا زندگی با شریک از {married['درصد بدون وزن']:.2f} درصد بدون وزن به {married['درصد وزندار']:.2f} درصد وزندار می‌رسد؛ بنابراین، تغییر اندک میانگین کلی خواب نباید باعث شود ترکیب گروه‌ها را در دو شیوه یکسان فرض کنیم.

## میانگین متغیرهای کمی

{markdown_table(formatted(numeric.drop(columns="نام متغیر")))}

سن و نسبت درآمد به مقادیر منتشرشده مربوط‌اند، زیرا مقدار 80 سن شامل افراد 80 ساله و بالاتر و مقدار 5 نسبت درآمد شامل مقادیر 5 و بیشتر است؛ بنابراین، میانگین این دو متغیر را نباید میانگین دقیق مقادیر بدون حد بالای انتشار دانست. میانگین خواب نیز با تقریب کدهای 2 و 14 محاسبه شده است و مقایسه دو نوع روز، به دلیل تفاوت مجموعه پاسخ‌های موجود، مقایسه جفت‌شده تغییر خواب هر فرد نیست.

میانگین سن ثبت‌شده نیز از {age['میانگین بدون وزن']:.2f} سال بدون وزن به {age['میانگین وزندار']:.2f} سال وزندار کاهش می‌یابد، که تفاوت میان ترکیب سنی نمونه و خلاصه حاصل از وزن‌دهی را نشان می‌دهد، هرچند محدودیت ثبت گروه 80 سال و بالاتر همچنان باقی است.

## میانگین خواب روزهای کاری در گروه‌ها

{markdown_table(formatted(groups.drop(columns="نام متغیر")))}

این جدول، رابطه توصیفی خواب با تحصیلات، جنس و تأهل را با و بدون وزن نشان می‌دهد، اما سایر ویژگی‌ها در آن ثابت نگه داشته نشده‌اند؛ ازاین‌رو، تفاوت میان گروه‌ها را نمی‌توان اثر مستقل تحصیلات، جنس یا تأهل دانست. میانگین سن و درآمد در جدول قبل نیز ترکیب پاسخ‌دهندگان را توصیف می‌کند و جایگزین بررسی رابطه این دو متغیر با خواب نیست.

برای گروه هرگز ازدواج‌نکرده، میانگین خواب پس از وزن‌دهی حدود {never_married['تفاوت وزن‌دهی؛ دقیقه']:.2f} دقیقه بیشتر می‌شود و این تغییر از تفاوت مشاهده‌شده در میانگین کلی خواب بزرگ‌تر است؛ بااین‌حال، بدون محاسبه عدم قطعیت نمی‌توان درباره معنی‌داری آماری آن نتیجه گرفت.

## حساسیت به دو گروه انتهایی خواب

{markdown_table(formatted(sensitivity.drop(columns="نام متغیر")))}

برای بررسی وابستگی نتایج به تقریب خواب، میانگین‌ها را بدون دو گروه «کمتر از 3 ساعت» و «14 ساعت و بیشتر» دوباره محاسبه کرده‌ایم، درحالی‌که این افراد از فایل خام یا سایر جدول‌ها حذف نشده‌اند. تغییر هر میانگین برابر مقدار بدون دو گروه منهای مقدار تقریب اصلی در همان گروه است و چون مجموعه مشاهدات تغییر می‌کند، این اختلاف را نباید مقدار معلوم خطای تقریب دانست.

## اطلاعات ناموجود و محدودیت تفسیر

{markdown_table(formatted(missing))}

درصدهای این جدول نسبت به همه {len(adults)} بزرگسال محاسبه شده‌اند، درحالی‌که مخرج درصدهای جدول ترکیب گروه‌ها فقط پاسخ‌های موجود هر متغیر بود. تفاوت سهم افراد از تعداد نمونه و از مجموع وزن نشان می‌دهد که گمشدگی نیز باید در هر دو مقیاس بررسی شود، اما وزن رسمی عدم پاسخ به سؤال مشخص را خودبه‌خود جبران نمی‌کند و این جدول نیز ثابت نمی‌کند که داده‌ها به‌طور تصادفی گمشده‌اند.

نتایج وزندار برای توجه به طرح نمونه‌گیری مناسب‌ترند، اما همچنان به اطلاعات موجود و فرض‌های ثبت‌شده درباره خواب و مقادیر انتهایی وابسته‌اند؛ همچنین، چون خطای استاندارد، فاصله اطمینان و آزمون فرض محاسبه نشده است، درباره معنی‌داری تفاوت‌ها نتیجه‌گیری نمی‌کنیم. متغیرهای طبقه و واحد نمونه‌گیری اولیه در مراحل استنباط لازم‌اند و وزن‌دهی حاضر نیز با تعدیل هم‌زمان متغیرها در رگرسیون متفاوت است.

## منابع و اجرای دوباره

- [راهنمای وزن‌دهی NHANES](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)
- [راهنمای متغیرهای جمعیت‌شناختی و وزن‌ها](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm)
- [راهنمای متغیرهای خواب](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SLQ_L.htm)

```sh
.venv/bin/python src/compare_all_weights.py
```
"""
    (OUT / "weighted_descriptive_comparisons.md").write_text(report, encoding="utf-8")
    weekday = numeric.loc[numeric["نام متغیر"].eq("SLD012")].iloc[0]
    education = groups.loc[groups["نام متغیر"].eq("DMDEDUC2")]
    summary = f"""# نقش وزن نمونه‌گیری در توصیف خواب

وزن‌دهی را در توزیع خواب روزهای کاری و میانگین خواب بر حسب تحصیلات بررسی می‌کنیم تا هم تغییر سهم مقادیر خواب و هم رابطه توصیفی آن با متغیر توضیحی اصلی مطالعه مشخص شود.

## روش و نتیجه اصلی

از وزن مصاحبه `WTINT2YR` استفاده کرده‌ایم که NCHS برای هر شرکت‌کننده منتشر کرده است، زیرا متغیرهای مطالعه از مصاحبه به دست آمده‌اند و نیازی به وزن معاینه ندارند. درصد وزندار هر مقدار خواب برابر سهم وزن پاسخ‌دهندگان آن مقدار از مجموع وزن پاسخ‌های موجود است و میانگین وزندار نیز از تقسیم مجموع حاصل‌ضرب وزن در خواب بر مجموع وزن همان افراد به دست می‌آید؛ بنابراین، افراد دو محاسبه در هر مقایسه یکسان‌اند و تفاوت نتیجه فقط به شیوه وزن‌دهی مربوط است.

در {int(weekday['تعداد پاسخ موجود'])} پاسخ موجود، میانگین تقریبی خواب روزهای کاری از {weekday['میانگین بدون وزن']:.3f} ساعت بدون وزن به {weekday['میانگین وزندار']:.3f} ساعت وزندار می‌رسد و تغییر آن {60*weekday['تفاوت در واحد متغیر']:.3f} دقیقه است. در پنج سطح تحصیلات، تغییر میانگین بر اثر وزن‌دهی از {education['تفاوت وزن‌دهی؛ دقیقه'].min():.2f} تا {education['تفاوت وزن‌دهی؛ دقیقه'].max():.2f} دقیقه است؛ بااین‌حال، کوچک‌بودن این تغییرها در مقایسه‌های فعلی به معنی بی‌نیازی از وزن‌دهی در مراحل بعد نیست.

## دو نمودار منتخب

![توزیع بدون وزن و وزندار خواب روزهای کاری](../../figures/phase1/weekday_sleep_distribution.png)

![میانگین بدون وزن و وزندار خواب بر حسب تحصیلات](../../figures/phase1/weekday_sleep_by_education.png)

در نمودار اول، دو گروه انتهایی با عنوان خود و به‌صورت هاشورخورده مشخص شده‌اند، اما میانگین‌های نمودار دوم کدهای 2 و 14 را به‌عنوان تقریب به کار می‌برند؛ به همین دلیل، [تحلیل حساسیت](weighted_sleep_comparison.md) و محاسبات درون گروه‌ها نیز در [گزارش پشتیبان](weighted_descriptive_comparisons.md) نگهداری شده‌اند. این مقایسه‌ها برای سن، جنس و تأهل تعدیل نشده‌اند و فاصله اطمینان یا آزمون فرض ندارند، درحالی‌که وزن رسمی نیز پاسخ‌های گمشده به سؤال مشخص را خودبه‌خود جبران نمی‌کند.

## منبع و بازتولید

مبنای انتخاب وزن، [راهنمای رسمی وزن‌دهی NHANES](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx) است؛ برای بازتولید محاسبات و نمودارها، به‌ترتیب دو دستور زیر را اجرا می‌کنیم:

```sh
.venv/bin/python src/compare_all_weights.py
.venv/bin/python src/plot_sleep_relationships.py
```
"""
    (OUT / "weighting_summary.md").write_text(summary, encoding="utf-8")
    print(formatted(categories).to_string(index=False))
    print(formatted(numeric).to_string(index=False))
    print(formatted(groups).to_string(index=False))


if __name__ == "__main__":
    main()
