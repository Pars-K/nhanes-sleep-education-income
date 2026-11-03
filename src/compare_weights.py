"""مقایسه میانگین بدون وزن و وزندار خواب روزهای کاری با وزن مصاحبه."""

from pathlib import Path

import numpy as np
import pandas as pd

from audit_adults import markdown_table
from describe_numeric import formatted


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase1"


def compare(data, label):
    values = data.SLD012.to_numpy()
    weights = data.WTINT2YR.to_numpy()
    if not len(data) or not np.isfinite(values).all():
        raise ValueError("مدت خواب باید موجود و متناهی باشد.")
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("وزن‌ها باید موجود، متناهی و مثبت باشند.")
    simple = float(np.mean(values))
    weighted = float(np.average(values, weights=weights))
    return {
        "روش برخورد با دو گروه انتهایی": label,
        "تعداد مشاهدات مشترک دو محاسبه": len(data),
        "میانگین بدون وزن؛ ساعت": simple,
        "میانگین وزندار؛ ساعت": weighted,
        "تفاوت وزندار منهای بدون وزن؛ دقیقه": 60 * (weighted - simple),
    }


def main():
    demo = pd.read_sas(ROOT / "data/raw/DEMO_L.xpt", format="xport", encoding=None)
    sleep = pd.read_sas(ROOT / "data/raw/SLQ_L.xpt", format="xport", encoding=None)
    if demo.SEQN.isna().any() or sleep.SEQN.isna().any():
        raise ValueError("شناسه گمشده در فایل خام وجود دارد.")
    merged = demo.merge(sleep, on="SEQN", how="left", validate="one_to_one")
    adults = merged.loc[merged.RIDAGEYR.ge(20)].copy()
    if adults.WTINT2YR.isna().any() or not np.isfinite(adults.WTINT2YR).all() or adults.WTINT2YR.le(0).any():
        raise ValueError("حداقل یک بزرگسال وزن مصاحبه معتبر ندارد.")
    if adults[["SDMVSTRA", "SDMVPSU"]].isna().any().any():
        raise ValueError("اطلاعات طرح نمونه‌گیری ناقص است.")
    available = adults.loc[adults.SLD012.notna()]
    restricted = available.loc[~available.SLD012.isin([2, 14])]
    main_result = compare(available, "کدهای 2 و 14 به‌عنوان تقریب")
    alternative = compare(restricted, "بدون دو گروه کمتر از 3 و 14 ساعت و بیشتر")
    comparison = pd.DataFrame([main_result, alternative])
    sensitivity = pd.DataFrame([
        ("بدون وزن", main_result["میانگین بدون وزن؛ ساعت"], alternative["میانگین بدون وزن؛ ساعت"],
         60 * (alternative["میانگین بدون وزن؛ ساعت"] - main_result["میانگین بدون وزن؛ ساعت"])),
        ("وزندار", main_result["میانگین وزندار؛ ساعت"], alternative["میانگین وزندار؛ ساعت"],
         60 * (alternative["میانگین وزندار؛ ساعت"] - main_result["میانگین وزندار؛ ساعت"])),
    ], columns=["محاسبه", "میانگین تقریب اصلی؛ ساعت", "میانگین بدون دو گروه انتهایی؛ ساعت", "تغییر میانگین؛ دقیقه"])
    missing = adults.SLD012.isna()
    missing_weight_share = 100 * adults.loc[missing, "WTINT2YR"].sum() / adults.WTINT2YR.sum()
    ends = available.SLD012.isin([2, 14])
    end_weight_share = 100 * available.loc[ends, "WTINT2YR"].sum() / available.WTINT2YR.sum()
    OUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT / "weighted_sleep_comparison.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT / "weighted_sleep_sensitivity.csv", index=False, encoding="utf-8-sig")
    weights = pd.DataFrame([
        ("تعداد بزرگسالان", len(adults)),
        ("تعداد وزن مصاحبه ناموجود یا نامعتبر", 0),
        ("کمترین وزن مصاحبه", f"{adults.WTINT2YR.min():.3f}"),
        ("بیشترین وزن مصاحبه", f"{adults.WTINT2YR.max():.3f}"),
        ("تعداد طبقه‌های نمونه‌گیری موجود", adults.SDMVSTRA.nunique()),
        ("تعداد ترکیب‌های طبقه و واحد نمونه‌گیری اولیه", len(adults[["SDMVSTRA", "SDMVPSU"]].drop_duplicates())),
    ], columns=["ویژگی", "مقدار"])
    weights.to_csv(OUT / "interview_weight_checks.csv", index=False, encoding="utf-8-sig")
    report = f"""# مقایسه میانگین بدون وزن و وزندار خواب

برای بررسی نقش طرح نمونه‌گیری، میانگین خواب روزهای کاری را در افراد 20 ساله و بالاتر یک بار با سهم یکسان برای هر فرد و بار دیگر با وزن مصاحبه محاسبه کرده‌ایم، درحالی‌که مجموعه مشاهدات در دو محاسبه یکسان است؛ بنابراین، تفاوت میانگین‌ها در هر ردیف جدول از وزن‌دهی ناشی می‌شود و با تغییر تعداد افراد مخلوط نشده است.

## دلیل انتخاب وزن

وزن `WTINT2YR` در فایل جمعیت‌شناختی توسط NCHS منتشر شده است و آن را بر اساس نتیجه خواب یا ترجیح تحلیلی خود تعیین نکرده‌ایم. این وزن‌ها با توجه به احتمال انتخاب افراد، عدم پاسخ در سطح پیمایش و تطبیق با تعداد جمعیت ساخته می‌شوند و از آنجا که خواب و متغیرهای فعلی مطالعه از مصاحبه به دست آمده‌اند، وزن مصاحبه را به‌جای وزن معاینه به کار برده‌ایم. به این ترتیب، میانگین بدون وزن ویژگی پاسخ‌دهندگان نمونه را توصیف می‌کند، درحالی‌که میانگین وزندار برای برآورد ویژگی جامعه هدف مناسب‌تر است، هرچند محدودیت اطلاعات گمشده و تقریب مدت خواب همچنان باقی می‌ماند.

## روش محاسبه و بررسی وزن‌ها

میانگین بدون وزن از تقسیم مجموع مدت خواب بر تعداد افراد و میانگین وزندار از تقسیم مجموع حاصل‌ضرب وزن در مدت خواب بر مجموع وزن همان افراد به دست آمده است:

$$
\\bar{{x}}_w = \\frac{{\\sum_i w_i x_i}}{{\\sum_i w_i}}
$$

در این رابطه، $x_i$ مدت خواب کدگذاری‌شده و $w_i$ وزن مصاحبه فرد است؛ وزن‌های منتشرشده تغییر نکرده‌اند و مخرج محاسبه فقط به افرادی مربوط است که در همان خلاصه وارد شده‌اند. وزن بزرگسالان و موجودبودن متغیرهای طبقه و واحد نمونه‌گیری اولیه نیز بررسی شده‌اند و هیچ فردی به دلیل وزن نامعتبر کنار گذاشته نشده است.

{markdown_table(weights)}

متغیرهای `SDMVSTRA` و `SDMVPSU` برای محاسبه درست خطای استاندارد، فاصله اطمینان و آزمون‌های مراحل بعد لازم‌اند، اما در فرمول این میانگین وزندار مستقیماً وارد نمی‌شوند. همچنین، وزن‌دهی نمونه‌گیری با تعدیل رابطه‌ها برای سن، جنس یا تأهل در رگرسیون متفاوت است و در این مرحله فاصله اطمینان یا آزمون فرض محاسبه نکرده‌ایم.

## مقایسه میانگین‌ها

{markdown_table(formatted(comparison))}

در محاسبه اصلی، میانگین بدون وزن {main_result['میانگین بدون وزن؛ ساعت']:.3f} ساعت و میانگین وزندار {main_result['میانگین وزندار؛ ساعت']:.3f} ساعت است که اختلاف آن‌ها {main_result['تفاوت وزندار منهای بدون وزن؛ دقیقه']:.3f} دقیقه می‌شود. این اختلاف نشان می‌دهد سهم یکسان افراد و سهم متناسب با وزن مصاحبه، خلاصه یکسانی از داده‌ها به دست نمی‌دهند، اما برای نسبت‌دادن این تفاوت به یک ویژگی مشخص، مانند سن یا جنس، به مقایسه‌های جداگانه نیاز داریم و از این جدول به‌تنهایی چنین نتیجه‌ای نمی‌گیریم.

## بررسی حساسیت به دو گروه انتهایی

{markdown_table(formatted(sensitivity))}

در ردیف اصلی، کدهای 2 و 14 خواب به‌عنوان تقریب دو گروه «کمتر از 3 ساعت» و «14 ساعت و بیشتر» به کار رفته‌اند و در محاسبه دوم، {int(ends.sum())} فرد این دو گروه فقط از همان خلاصه کنار گذاشته شده‌اند؛ بنابراین، محاسبه دوم گروه محدودتری را توصیف می‌کند و مدت واقعی خواب افراد گروه‌های انتهایی را مشخص نمی‌کند. سهم این دو گروه از تعداد پاسخ‌های موجود {100 * ends.mean():.2f} درصد و سهم آن‌ها از مجموع وزن پاسخ‌های موجود {end_weight_share:.2f} درصد است، که نشان می‌دهد فراوانی نمونه و سهم وزنی را باید جداگانه تفسیر کنیم.

## اطلاعات ناموجود و دامنه تفسیر

از {len(adults)} بزرگسال، مدت خواب روزهای کاری برای {int(missing.sum())} نفر موجود نیست؛ این افراد {100 * missing.mean():.2f} درصد نمونه بزرگسالان و {missing_weight_share:.2f} درصد مجموع وزن آن‌ها را تشکیل می‌دهند. وزن رسمی، عدم پاسخ در سطح مصاحبه را در نظر می‌گیرد، اما پاسخ گمشده به یک سؤال مشخص را به‌طور خودکار جبران نمی‌کند؛ ازاین‌رو، محاسبه میانگین با پاسخ‌های موجود همچنان ممکن است از تفاوت پاسخ‌دهندگان و افراد دارای اطلاعات ناموجود تأثیر بپذیرد.

میانگین وزندار نیز به مدت خواب کدگذاری‌شده مربوط است و خواب کل شبانه‌روز، از جمله چرت‌زدن، را لزوماً پوشش نمی‌دهد؛ علاوه بر این، نتیجه تحلیل حساسیت بدون دو گروه انتهایی را نباید برآورد میانگین خواب همه بزرگسالان دانست، زیرا گروه مورد توصیف در آن محدود شده است.

## منابع و اجرای دوباره

- [راهنمای وزن‌دهی NHANES](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)
- [متغیرهای جمعیت‌شناختی و وزن‌های این دوره](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm)
- [راهنمای متغیرهای خواب](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SLQ_L.htm)

```sh
.venv/bin/python src/compare_weights.py
```
"""
    (OUT / "weighted_sleep_comparison.md").write_text(report, encoding="utf-8")
    print(formatted(comparison).to_string(index=False))
    print(formatted(sensitivity).to_string(index=False))
    print(f"سهم وزنی اطلاعات ناموجود: {missing_weight_share:.3f} درصد")
    print(f"سهم وزنی دو گروه انتهایی از پاسخ‌های موجود: {end_weight_share:.3f} درصد")


if __name__ == "__main__":
    main()
