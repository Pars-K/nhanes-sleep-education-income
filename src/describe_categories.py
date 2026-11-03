"""جدول فراوانی متغیرهای کیفی با استفاده از پاسخ‌های موجود هر متغیر."""

from pathlib import Path

import pandas as pd

from audit_adults import markdown_table


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase1"
CATEGORIES = {
    "RIAGENDR": ("جنس", {1: "مرد", 2: "زن"}, []),
    "DMDEDUC2": ("تحصیلات", {
        1: "کمتر از پایه نهم",
        2: "پایه نهم تا یازدهم یا پایه دوازدهم بدون دیپلم",
        3: "دیپلم دبیرستان یا معادل آن (GED)",
        4: "تحصیلات دانشگاهی ناتمام یا مدرک کاردانی",
        5: "کارشناسی و بالاتر",
    }, [7, 9]),
    "DMDMARTZ": ("وضعیت تأهل", {
        1: "متأهل یا زندگی با شریک",
        2: "بیوه، مطلقه یا جداشده",
        3: "هرگز ازدواج‌نکرده",
    }, [77, 99]),
}


def main():
    demo = pd.read_sas(ROOT / "data/raw/DEMO_L.xpt", format="xport", encoding=None)
    adults = demo.loc[demo.RIDAGEYR.ge(20)]
    rows = []
    sections = []
    for col, (label, labels, special) in CATEGORIES.items():
        values = adults[col]
        unexpected = values.notna() & ~values.isin([*labels, *special])
        if unexpected.any():
            raise ValueError(f"کد تعریف‌نشده در {col}")
        valid = values.isin(labels)
        n = int(valid.sum())
        missing = int((~valid).sum())
        if n == 0:
            raise ValueError(f"پاسخ موجود برای {col} یافت نشد.")
        table_rows = []
        for code, category in labels.items():
            count = int(values.eq(code).sum())
            percentage = 100 * count / n
            rows.append((col, label, code, category, count, n, percentage))
            table_rows.append((category, count, f"{percentage:.2f}"))
        frame = pd.DataFrame(table_rows, columns=["گروه", "فراوانی", "درصد پاسخ‌های موجود"])
        sections.append(
            f"## {label}\n\n"
            f"برای این متغیر، پاسخ {n} نفر موجود است و اطلاعات {missing} نفر "
            f"({100 * missing / len(adults):.2f} درصد از {len(adults)} نفر) در دسترس نیست؛ "
            "بنابراین، درصد گروه‌های جدول بر اساس پاسخ‌های موجود محاسبه شده است.\n\n"
            + markdown_table(frame)
        )
        if sum(row[1] for row in table_rows) != n:
            raise ValueError("مجموع فراوانی‌ها با تعداد پاسخ‌های موجود برابر نیست.")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=[
        "نام متغیر", "متغیر", "کد", "گروه", "فراوانی", "تعداد پاسخ‌های موجود", "درصد پاسخ‌های موجود",
    ]).to_csv(OUT / "categorical_frequencies.csv", index=False, encoding="utf-8-sig")
    report = """# توزیع فراوانی متغیرهای کیفی

این جدول‌ها توزیع فراوانی متغیرهای کیفی را در افراد 20 ساله و بالاتر نشان می‌دهند و از آنجا که محاسبات بدون وزن نمونه‌گیری انجام شده‌اند، درصدهای حاصل به نمونه مربوط‌اند و برآورد درصدهای جامعه آمریکا نیستند. برای هر متغیر، پاسخ‌های موجود مبنای درصد گروه‌ها قرار گرفته‌اند، درحالی‌که کدهای امتناع از پاسخ و ندانستن پاسخ در شمار اطلاعات ناموجود منظور شده‌اند و داده‌های خام نیز بدون تغییر باقی مانده‌اند. همچنین، به دلیل گردکردن درصدها، ممکن است مجموع مقادیر نمایش‌داده‌شده اندکی با 100 تفاوت داشته باشد.

""" + "\n\n".join(sections) + """

## برداشت اولیه

در میان پاسخ‌های موجود، گروه کارشناسی و بالاتر بیشترین فراوانی را در تحصیلات دارد و پس از آن، گروه تحصیلات دانشگاهی ناتمام یا مدرک کاردانی قرار می‌گیرد؛ همچنین، در وضعیت تأهل، فراوانی گروه متأهل یا زندگی با شریک از هر یک از دو گروه دیگر بیشتر است. بااین‌حال، این جدول‌ها تنها ترکیب نمونه را از نظر متغیرهای کیفی توصیف می‌کنند و برای بررسی رابطه تحصیلات یا وضعیت تأهل با مدت خواب، لازم است در مرحله بعد اطلاعات این متغیرها را در کنار یکدیگر قرار دهیم.

## اجرای دوباره

```sh
.venv/bin/python src/describe_categories.py
```
"""
    (OUT / "categorical_frequencies.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
