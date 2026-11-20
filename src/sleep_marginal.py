"""Build the first phase 2 table from the two published phase 1 summaries."""

import csv
from math import fsum, isclose, sqrt
from pathlib import Path


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def build_report(phase1):
    counts = [row for row in read_table(phase1 / "sleep_distribution_frequencies.csv")
              if row["متغیر"] == "SLD012"]
    weights = read_table(phase1 / "weekday_distribution_weight_comparison.csv")
    by_code = {row["کد منتشرشده"]: row for row in weights}
    codes = {row["کد منتشرشده"] for row in counts}
    if len(codes) != len(counts) or len(by_code) != len(weights) or codes != by_code.keys():
        raise ValueError("Sleep categories do not match between phase 1 tables.")
    total = sum(int(row["فراوانی"]) for row in counts)
    weighted_total = sum(float(row["درصد وزندار"]) for row in weights)
    if not isclose(weighted_total, 100):
        raise ValueError("Weighted percentages must sum to 100.")

    lines = [
        "# توزیع حاشیه‌ای مدت خواب در روزهای کاری",
        "",
        "جامعه هدف شامل افراد 20 ساله و بالاتر از جمعیت غیرنظامی و غیرمؤسسه‌ای آمریکا در دوره اوت 2021 تا اوت 2023 است. برای فردی که از این جمعیت انتخاب می‌شود، X مدت خواب معمول او در روزهای کاری بر حسب ساعت است. مقدار مشاهده‌شده از متغیر `SLD012` در NHANES به دست می‌آید. خواب در مقیاس پیوسته تعریف می‌شود و داده در گام‌های نیم‌ساعته و با دو رده باز در انتهای دامنه منتشر شده است.",
        "",
        "این جدول توزیع خواب ثبت‌شده را در کل نمونه بزرگسالان، پیش از گروه‌بندی بر اساس تحصیلات یا نسبت درآمد خانواده به خط فقر، نشان می‌دهد. رده‌های انتهایی با عنوان منتشرشده محاسبه شده‌اند.",
        "",
        f"از 7809 بزرگسال، {total} نفر برای خواب روز کاری پاسخ موجود داشتند. درصد بدون وزن هر رده برابر است با 100 برابر تعداد پاسخ‌های رده تقسیم بر {total}. درصد وزندار برابر است با 100 برابر مجموع وزن مصاحبه (`WTINT2YR`) در رده تقسیم بر مجموع وزن همان {total} پاسخ‌دهنده. در نتیجه هر سطر یک احتمال تجربی را در مقیاس درصد نشان می‌دهد.",
        "",
        "| خواب ثبت‌شده (ساعت) | تعداد پاسخ | درصد بدون وزن | درصد وزندار |",
        "|---|---:|---:|---:|",
    ]
    sample_mass = {}
    survey_mass = {}
    for row in sorted(counts, key=lambda r: float(r["کد منتشرشده"])):
        other = by_code[row["کد منتشرشده"]]
        count = int(row["فراوانی"])
        label = row["عنوان مقدار یا گروه"]
        percent = 100 * count / total
        if label != other["عنوان مقدار یا گروه"] or not isclose(percent, float(other["درصد بدون وزن"])):
            raise ValueError(f"Inconsistent phase 1 summaries for {label}")
        weighted_percent = float(other["درصد وزندار"])
        hours = float(row["کد منتشرشده"])
        sample_mass[hours] = count / total
        survey_mass[hours] = weighted_percent / 100
        lines.append(f"| {label} | {count} | {percent:.2f} | {weighted_percent:.2f} |")
    lines += [
        f"| کل پاسخ‌های موجود | {total} | 100.00 | 100.00 |",
        "",
        "درصدهای هر رده برای نمایش تا دو رقم اعشار گرد شده‌اند. منابع محاسبه، [فراوانی پاسخ‌ها](../phase1/sleep_distribution_frequencies.csv) و [درصدهای وزندار](../phase1/weekday_distribution_weight_comparison.csv) در فاز اول هستند.",
    ]
    lines += [
        "",
        "## امید ریاضی و واریانس مدت خواب روزهای کاری",
        "",
        "در این محاسبه، برای رده «کمتر از 3 ساعت» مقدار 2 و برای «14 ساعت و بیشتر» مقدار 14 را مانند فاز اول به‌عنوان تقریب به کار بردیم. نتایج به مدت خواب کدگذاری‌شده مربوط‌اند. برای هر رده، احتمال p از فراوانی یا سهم وزن مصاحبه در میان پاسخ‌های موجود به دست آمد. محاسبات با احتمال‌های پیش از گردکردن انجام شدند.",
        "",
        "امید ریاضی از جمع حاصل‌ضرب ساعت هر رده در احتمال آن به دست می‌آید. واریانس، میانگین مربع فاصله از همین امید ریاضی است. برای هر ستون احتمال، مرکز همان توزیع را مبنا گرفتیم.",
        "",
        "$$\\mu=\\sum_i x_i p_i, \\qquad \\operatorname{Var}(X)=\\sum_i (x_i-\\mu)^2p_i.$$",
        "",
        "| توزیع تجربی | امید ریاضی (ساعت) | واریانس (ساعت‌مربع) | انحراف معیار (ساعت) |",
        "|---|---:|---:|---:|",
    ]
    for label, mass in [("بدون وزن", sample_mass), ("با وزن مصاحبه", survey_mass)]:
        mean = fsum(x * p for x, p in mass.items())
        variance = fsum((x - mean) ** 2 * p for x, p in mass.items())
        lines.append(f"| {label} | {mean:.4f} | {variance:.4f} | {sqrt(variance):.4f} |")
    lines += [
        "",
        f"در سطر بدون وزن هر یک از {total} نفر سهم یکسان دارد. واریانس توزیع تجربی مخرج {total} دارد، در حالی که در فاز اول برای واریانس نمونه از مخرج {total - 1} استفاده شد. ضرب واریانس تجربی بدون وزن در نسبت {total}/{total - 1}، واریانس نمونه فاز اول را بازمی‌سازد.",
        "",
        "انحراف معیار پراکندگی خواب بین پاسخ‌دهندگان را بر حسب ساعت نشان می‌دهد. در سطر وزندار، وزن‌های مصاحبه سهم افراد را در توزیع تغییر می‌دهند. محاسبه خطای استاندارد میانگین به اطلاعات طرح نمونه‌گیری نیز نیاز دارد.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    reports = Path(__file__).resolve().parents[1] / "reports"
    report = build_report(reports / "phase1")
    output = reports / "phase2" / "sleep_marginal.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report, encoding="utf-8")
