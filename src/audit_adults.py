"""بررسی داده‌های افراد 20 ساله و بالاتر، بدون جایگزینی مقادیر یا حذف اطلاعات ناقص."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase1"
VARIABLES = {
    "RIDAGEYR": "سن",
    "RIAGENDR": "جنس",
    "DMDEDUC2": "تحصیلات",
    "DMDMARTZ": "وضعیت تأهل",
    "INDFMPIR": "نسبت درآمد به خط فقر",
    "SLD012": "خواب روزهای کاری",
    "SLD013": "خواب روزهای غیرکاری",
}
SPECIAL = {"DMDEDUC2": [7, 9], "DMDMARTZ": [77, 99]}


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def main():
    demo = pd.read_sas(ROOT / "data/raw/DEMO_L.xpt", format="xport", encoding=None)
    sleep = pd.read_sas(ROOT / "data/raw/SLQ_L.xpt", format="xport", encoding=None)
    for table in (demo, sleep):
        if table.SEQN.isna().any() or not table.SEQN.is_unique:
            raise ValueError("شناسه‌ها باید موجود و یکتا باشند.")
    if not set(sleep.SEQN).issubset(set(demo.SEQN)):
        raise ValueError("در فایل خواب شناسه‌ای بدون تطابق وجود دارد.")
    merged = demo.merge(sleep, on="SEQN", how="left", validate="one_to_one", indicator=True)
    adults = merged.loc[merged.RIDAGEYR.ge(20)].copy()
    n = len(adults)
    if n == 0:
        raise ValueError("نمونه بزرگسالان خالی است.")

    flow = pd.DataFrame([
        ("فایل جمعیت‌شناختی پیش از ادغام", len(demo)),
        ("فایل خواب پیش از ادغام", len(sleep)),
        ("پس از اتصال از سمت فایل جمعیت‌شناختی", len(merged)),
        ("افراد زیر 20 سال؛ خارج از جامعه مورد مطالعه", int(merged.RIDAGEYR.lt(20).sum())),
        ("افراد با سن گمشده", int(merged.RIDAGEYR.isna().sum())),
        ("افراد 20 ساله و بالاتر", n),
        ("بزرگسالان دارای رکورد متناظر خواب", int(adults["_merge"].eq("both").sum())),
        ("بزرگسالان فاقد رکورد متناظر خواب", int(adults["_merge"].eq("left_only").sum())),
    ], columns=["مرحله", "تعداد"])

    rows = []
    available = {}
    for col, label in VARIABLES.items():
        blank = adults[col].isna()
        special = adults[col].isin(SPECIAL.get(col, []))
        unavailable = blank | special
        available[col] = ~unavailable
        rows.append((label, col, int(blank.sum()), int(special.sum()),
                     int(unavailable.sum()), f"{100 * unavailable.mean():.2f}"))
    missing = pd.DataFrame(rows, columns=[
        "متغیر", "نام ستون", "مقدار گمشده در فایل", "کد عدم پاسخ",
        "مجموع اطلاعات ناموجود", "درصد از کل بزرگسالان",
    ])

    endpoints = pd.DataFrame([
        ("خواب روزهای کاری کمتر از 3 ساعت؛ کد 2", int(adults.SLD012.eq(2).sum())),
        ("خواب روزهای کاری 14 ساعت و بیشتر؛ کد 14", int(adults.SLD012.eq(14).sum())),
        ("خواب روزهای غیرکاری کمتر از 3 ساعت؛ کد 2", int(adults.SLD013.eq(2).sum())),
        ("خواب روزهای غیرکاری 14 ساعت و بیشتر؛ کد 14", int(adults.SLD013.eq(14).sum())),
        ("سن 80 سال و بالاتر", int(adults.RIDAGEYR.eq(80).sum())),
        ("نسبت درآمد 5 و بیشتر", int(adults.INDFMPIR.eq(5).sum())),
    ], columns=["گروه", "تعداد"])

    allowed_sleep = [2, 14, *[v / 2 for v in range(6, 28)]]
    allowed = {
        "RIDAGEYR": adults.RIDAGEYR.between(20, 80) & adults.RIDAGEYR.mod(1).eq(0),
        "RIAGENDR": adults.RIAGENDR.isin([1, 2]),
        "DMDEDUC2": adults.DMDEDUC2.isin([1, 2, 3, 4, 5, 7, 9]),
        "DMDMARTZ": adults.DMDMARTZ.isin([1, 2, 3, 77, 99]),
        "INDFMPIR": adults.INDFMPIR.between(0, 5),
        "SLD012": adults.SLD012.isin(allowed_sleep),
        "SLD013": adults.SLD013.isin(allowed_sleep),
        "WTINT2YR": adults.WTINT2YR.gt(0),
        "SDMVSTRA": adults.SDMVSTRA.isin(range(173, 188)),
        "SDMVPSU": adults.SDMVPSU.isin([1, 2]),
    }
    checks = pd.DataFrame([
        (col, int((adults[col].notna() & ~ok).sum()), int(adults[col].isna().sum()))
        for col, ok in allowed.items()
    ], columns=["متغیر", "مقدار خارج از دامنه یا کدهای مجاز", "مقدار گمشده"])

    main_cols = ["SLD012", "DMDEDUC2", "INDFMPIR", "RIDAGEYR", "RIAGENDR", "DMDMARTZ"]
    complete = pd.DataFrame({col: available[col] for col in main_cols}).all(axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [("sample_flow", flow), ("missingness_adults", missing),
                        ("grouped_values", endpoints), ("range_checks", checks)]:
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")

    text = f"""# بررسی داده‌های بزرگسالان

با توجه به محدوده سنی مطالعه که افراد 20 ساله و بالاتر را در بر می‌گیرد، فایل خواب را از طریق شناسه `SEQN` و به‌صورت یک‌به‌یک به فایل جمعیت‌شناختی متصل کردیم و سپس افراد واجد شرط سنی را مشخص کردیم. در این مرحله، فایل‌های خام بدون تغییر باقی مانده‌اند و به‌جز شرط سنی، محدودیت دیگری برای ورود به نمونه اعمال نشده است؛ بنابراین، هیچ فردی صرفاً به دلیل اطلاعات ناقص کنار گذاشته نشده است.

حد پایین سن را 20 سال در نظر گرفتیم، زیرا اطلاعات تحصیلات بزرگسالان و وضعیت تأهل از این سن در دسترس است؛ همچنین، اتصال را از سمت فایل جمعیت‌شناختی انجام دادیم تا افراد فاقد رکورد خواب پیش از شمارش و بررسی علت نبود اطلاعات کنار گذاشته نشوند.

## تعداد مشاهدات

{markdown_table(flow)}

از آنجا که هر ردیف جدول بالا یک شمارش مستقل یا مرحله مشخص از انتخاب نمونه را نشان می‌دهد، اعداد ردیف‌ها را نباید با هم جمع کرد. همچنین، اگرچه تمام افراد واجد شرط سنی رکورد متناظری در فایل خواب دارند، وجود این رکورد لزوماً به معنی کامل‌بودن پاسخ همه متغیرها نیست و به همین دلیل، اطلاعات ناموجود را جداگانه بررسی می‌کنیم.

## اطلاعات ناموجود

{markdown_table(missing)}

کدهای 7 و 9 در تحصیلات و کدهای 77 و 99 در وضعیت تأهل به معنی امتناع از پاسخ یا ندانستن پاسخ‌اند و ازاین‌رو، در جدول بالا در شمار اطلاعات ناموجود قرار گرفته‌اند، هرچند مقدار ذخیره‌شده آن‌ها تغییر نکرده است. برای اینکه میزان اطلاعات ناموجود متغیرها قابل مقایسه باشد، مخرج همه درصدها {n} نفر در نظر گرفته شده و محاسبات نیز بدون وزن نمونه‌گیری انجام شده‌اند.

در بررسی هم‌زمان متغیرهای مدل اصلی، {int(complete.sum())} نفر برای همه متغیرها اطلاعات موجود دارند، درحالی‌که {int((~complete).sum())} نفر دست‌کم در یک متغیر اطلاعات ناموجود دارند؛ بنابراین، استفاده از تحلیل موارد کامل می‌تواند تعداد مشاهدات را کاهش دهد. این شمارش تنها برای شناخت پیامد احتمالی آن روش انجام شده و به معنی انتخاب نهایی آن نیست؛ همچنین، مقادیر گروه‌بندی‌شده خواب در این مرحله اطلاعات موجود به حساب آمده‌اند.

## مقادیر ثبت‌شده به‌صورت گروهی

{markdown_table(endpoints)}

این مقادیر داده گمشده نیستند، بلکه اطلاعاتی را به‌صورت گروهی در اختیار می‌گذارند؛ برای مثال، کد 2 مدت خواب نشان‌دهنده کمتر از 3 ساعت است و نباید آن را دقیقاً 2 ساعت تفسیر کرد. به همین دلیل، در جدول‌های فراوانی عنوان گروه‌ها را حفظ می‌کنیم، اما در خلاصه‌های عددی، کدهای منتشرشده 2 و 14 را به‌عنوان تقریب به کار می‌بریم و نتیجه را با محاسبه دوباره بدون دو گروه انتهایی مقایسه می‌کنیم. در نتیجه، میانگین اصلی به متغیر کدگذاری‌شده مربوط است و میانگین دقیق زمان واقعی خواب نیست، درحالی‌که محاسبه دوم گروه محدودتری از افراد را توصیف می‌کند و به‌تنهایی نیز درستی تقریب را ثابت نمی‌کند.

## بررسی دامنه و کدهای مجاز

{markdown_table(checks)}

هرچند این بررسی نشان می‌دهد که مقادیر با دامنه و کدهای منتشرشده سازگارند، از آن نمی‌توان نتیجه گرفت که داده‌ها عاری از خطای اندازه‌گیری یا پاسخ‌گویی هستند. از نظر اطلاعات طرح نمونه‌گیری نیز همه بزرگسالان وزن مصاحبه مثبت دارند و اطلاعات طبقه و واحد نمونه‌گیری اولیه آن‌ها موجود است؛ بنابراین، این متغیرها را می‌توان در مراحل بعدی تحلیل به کار برد.

## روش محاسبه آمار توصیفی

برای تهیه جدول‌های توصیفی از پاسخ‌های موجود هر متغیر استفاده می‌کنیم و در کنار آن، تعداد اطلاعات ناموجود و مخرج درصدها را نیز نشان می‌دهیم، به‌طوری‌که درصد گروه‌ها در میان پاسخ‌های موجود و درصد اطلاعات ناموجود نسبت به کل نمونه محاسبه می‌شود. در مقایسه چند متغیر نیز تنها وجود اطلاعات همان متغیرها لازم است و داده‌های گمشده با صفر، میانگین یا مقدار دیگری جایگزین نمی‌شوند.

این روش را برای جدول‌های توصیفی انتخاب کردیم، زیرا نسبت درآمد به خط فقر برای 16.9 درصد بزرگسالان موجود نیست و کنارگذاشتن همه این افراد از تمام جدول‌ها، اطلاعات قابل استفاده آن‌ها درباره خواب، تحصیلات و سایر متغیرها را نیز از دسترس تحلیل خارج می‌کند.

این روش توصیفی از داده‌های مشاهده‌شده به دست می‌دهد، اما اگر افراد دارای اطلاعات ناموجود از نظر ویژگی‌های مرتبط با مطالعه با سایر افراد تفاوت داشته باشند، نتایج ممکن است نماینده تمام جامعه آماری نباشند؛ ازاین‌رو، انتخاب روش رگرسیون را به بررسی بیشتر الگوی گمشدگی و پیامدهای آن موکول می‌کنیم.

## منابع و روش اجرا

- [راهنمای متغیرهای جمعیت‌شناختی](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm)
- [راهنمای متغیرهای خواب](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SLQ_L.htm)

برای تولید مجدد جدول‌ها و این گزارش، از پوشه اصلی پروژه دستور زیر را اجرا می‌کنیم:

```sh
.venv/bin/python src/audit_adults.py
```
"""
    (OUT / "adult_data_audit.md").write_text(text, encoding="utf-8")
    print(flow.to_string(index=False))
    print(missing.to_string(index=False))
    print(endpoints.to_string(index=False))
    print(f"اطلاعات موجود برای تمام متغیرهای مدل اصلی: {int(complete.sum())}")
    print(f"تعداد مقادیر خارج از دامنه یا کدهای مجاز: {checks.iloc[:, 1].sum()}")


if __name__ == "__main__":
    main()
