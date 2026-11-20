"""Build the unweighted sleep-by-education event table for phase 2."""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


def binomial_section(valid):
    group_size = 30
    short_sleep = valid.sort_values("SEQN").SLD012.lt(7).to_numpy()
    n = len(short_sleep)
    p = short_sleep.mean()
    groups, remainder = divmod(n, group_size)
    shuffled = np.random.default_rng(2026).permutation(short_sleep)
    counts = shuffled[:groups * group_size].reshape(groups, group_size).sum(axis=1)
    variance = group_size * p * (1 - p)
    lines = [
        "",
        "## شمارش خواب کمتر از 7 ساعت در یک گروه 30 نفره",
        "",
        f"متغیر تصادفی Y تعداد افرادی است که در یک گروه {group_size} نفره، خواب کمتر از 7 ساعت در روزهای کاری گزارش می‌کنند. اگر وضعیت خواب افراد مستقل باشد و هر انتخاب احتمال یکسان p برای این رویداد داشته باشد، Y توزیع دوجمله‌ای دارد. برای مقایسه با نمونه، p را از فراوانی بدون وزن همه افراد واجد شرایط برآورد کردیم ({short_sleep.sum()} از {n} نفر، برابر {p:.4f}).",
        "",
        f"برای مقایسه تجربی، افراد را به ترتیب شناسه مرتب کردیم، با بذر ثابت 2026 ترتیب آن‌ها را یک بار تصادفی کردیم و {groups} گروه جداگانه {group_size} نفره ساختیم. {remainder} نفر آخر در برآورد p حضور دارند و فقط از مقایسه گروه‌های هم‌اندازه کنار گذاشته شده‌اند.",
        "",
        "| شاخص شمارش Y | در گروه‌های ساخته‌شده | مدل دوجمله‌ای |",
        "|---|---:|---:|",
        f"| میانگین (نفر) | {counts.mean():.2f} | {group_size * p:.2f} |",
        f"| واریانس (نفر به توان دو) | {counts.var(ddof=1):.2f} | {variance:.2f} |",
        "",
        "میانگین و واریانس مدل به‌ترتیب از np و np(1−p)، با n برابر اندازه هر گروه، محاسبه شدند. واریانس تجربی از شمارش‌های گروه‌ها با مخرج تعداد گروه‌ها منهای یک به دست آمد.",
        "",
        "| تعداد افراد با خواب کمتر از 7 ساعت در هر گروه | تعداد گروه‌ها | سهم مشاهده‌شده (درصد) | احتمال دوجمله‌ای (درصد) |",
        "|---|---:|---:|---:|",
    ]
    for low, high in [(0, 3), (4, 8), (9, group_size)]:
        observed_groups = ((low <= counts) & (counts <= high)).sum()
        probability = sum(comb(group_size, k) * p**k * (1 - p)**(group_size - k)
                          for k in range(low, high + 1))
        lines.append(f"| {low} تا {high} | {observed_groups} | {100 * observed_groups / groups:.2f} | {100 * probability:.2f} |")
    lines.extend([
        "",
        f"چون افراد بدون جایگذاری از فهرست ثابتی انتخاب می‌شوند، توزیع دقیق شمارش یک گروه تصادفی در این تمرین فوق‌هندسی است. انتخاب {group_size} نفر از {n} نفر سهم کوچکی از فهرست است و ضریب تصحیح واریانس جامعه محدود برابر {(n - group_size) / (n - 1):.4f} است. بنابراین برای یک گروه تصادفی در این فهرست، دوجمله‌ای تقریب نزدیکی به دست می‌دهد.",
        "",
        "در این تقسیم، تقریباً همه افراد همان نمونه در گروه‌ها حضور دارند، بنابراین نزدیک بودن میانگین گروه‌ها به میانگین مدل تا حد زیادی از روش ساخت گروه‌ها ناشی می‌شود. پراکندگی و سهم بازه‌ها رفتار شمارش‌ها را در یک تقسیم تصادفی نشان می‌دهند. طرح واقعی NHANES خوشه‌بندی و وزن‌های نابرابر دارد و استقلال افراد در آن با این تمرین ارزیابی نمی‌شود.",
    ])
    return lines


def build_report(root):
    raw = root / "data/raw"
    demo = pd.read_sas(raw / "DEMO_L.xpt", format="xport")[["SEQN", "RIDAGEYR", "DMDEDUC2"]]
    sleep = pd.read_sas(raw / "SLQ_L.xpt", format="xport")[["SEQN", "SLD012"]]
    adults = demo.loc[demo.RIDAGEYR.ge(20)].merge(sleep, on="SEQN", how="left", validate="one_to_one")
    has_sleep = adults.SLD012.notna()
    valid = adults.loc[has_sleep & adults.DMDEDUC2.isin([1, 2, 3, 4, 5])]
    table = pd.crosstab(valid.DMDEDUC2.eq(5), valid.SLD012.lt(7)).reindex(
        index=[True, False], columns=[True, False], fill_value=0
    )
    reference = pd.read_csv(root / "reports/phase1/sleep_by_group_summaries.csv")
    education = reference.loc[reference["متغیر"].eq("تحصیلات")]
    degree_count = education.loc[education["گروه"].eq("کارشناسی و بالاتر"), "تعداد"].item()
    if table.to_numpy().sum() != education["تعداد"].sum() or table.iloc[0].sum() != degree_count:
        raise ValueError("Education margins do not match the phase 1 summaries.")

    lines = [
        "# رویدادهای خواب و تحصیلات در روزهای کاری",
        "",
        "واحد مشاهده یک فرد 20 ساله یا بالاتر است. متغیر X همان [مدت خواب معمول در روزهای کاری](sleep_marginal.md) است. برای بررسی احتمال مشترک و شرطی، دو رویداد زیر را از پیش تعریف کردیم.",
        "",
        "- رویداد A هنگامی رخ می‌دهد که مدت خواب ثبت‌شده (`SLD012`) کمتر از 7 ساعت باشد. مقدار 7 در مکمل A قرار دارد. رده «کمتر از 3 ساعت» در A و رده «14 ساعت و بیشتر» در مکمل آن قرار می‌گیرند.",
        "- رویداد B هنگامی رخ می‌دهد که بالاترین سطح تحصیلات فرد کارشناسی یا بالاتر باشد (کد 5 در `DMDEDUC2`). کدهای معتبر 1 تا 4 مکمل B را تشکیل می‌دهند.",
        "",
        f"از {len(adults)} بزرگسال، {has_sleep.sum()} نفر پاسخ خواب داشتند. از این افراد، {has_sleep.sum() - len(valid)} نفر پاسخ معتبر تحصیلات نداشتند. جدول مشترک شامل {len(valid)} نفر با هر دو پاسخ معتبر است. هر نفر یک بار شمرده شده است.",
        "",
        "| تحصیلات | خواب کمتر از 7 ساعت | خواب 7 ساعت یا بیشتر | جمع |",
        "|---|---:|---:|---:|",
    ]
    for label, counts in zip(["کارشناسی یا بالاتر", "پایین‌تر از کارشناسی", "جمع"],
                             [table.iloc[0], table.iloc[1], table.sum()]):
        lines.append(f"| {label} | {counts.iloc[0]} | {counts.iloc[1]} | {counts.sum()} |")
    lines.extend(["", "اعداد جدول فراوانی بدون وزن افراد در نمونه‌اند."])
    total = len(valid)
    lines.extend([
        "",
        "## احتمال‌های حاشیه‌ای و مشترک",
        "",
        f"اگر از میان {total} نفر دارای هر دو پاسخ معتبر یک نفر را با احتمال برابر انتخاب کنیم، احتمال تجربی هر رویداد از تقسیم تعداد افراد دارای آن ویژگی بر {total} به دست می‌آید.",
        "",
        "| رویداد | صورت | مخرج | احتمال تجربی (درصد) |",
        "|---|---:|---:|---:|",
    ])
    events = [
        ("P(A)، خواب کمتر از 7 ساعت", table[True].sum()),
        ("P(B)، تحصیلات کارشناسی یا بالاتر", table.loc[True, :].sum()),
        ("P(A∩B)، داشتن هر دو ویژگی", table.at[True, True]),
    ]
    for label, count in events:
        lines.append(f"| {label} | {count} | {total} | {100 * count / total:.2f} |")
    lines.extend([
        "",
        "احتمال‌های حاشیه‌ای سهم هر ویژگی را در کل نمونه تحلیلی نشان می‌دهند. احتمال مشترک سهم افرادی از همین نمونه است که هم تحصیلات کارشناسی یا بالاتر دارند و هم خواب کمتر از 7 ساعت در روزهای کاری گزارش کرده‌اند.",
    ])
    group_sizes = table.sum(axis=1)
    short_sleep_pct = 100 * table[True] / group_sizes
    difference = short_sleep_pct.at[False] - short_sleep_pct.at[True]
    lines.extend([
        "",
        "## احتمال شرطی خواب کمتر از 7 ساعت بر حسب تحصیلات",
        "",
        "برای هر سطح از گروه‌بندی تحصیلات، تعداد افراد با خواب کمتر از 7 ساعت را بر تعداد کل افراد همان گروه با پاسخ معتبر تقسیم کردیم.",
        "",
        "| گروه تحصیلی | تعداد با خواب کمتر از 7 ساعت | تعداد افراد گروه | احتمال شرطی (درصد) |",
        "|---|---:|---:|---:|",
    ])
    for degree, label in [(True, "کارشناسی یا بالاتر"), (False, "پایین‌تر از کارشناسی")]:
        lines.append(f"| {label} | {table.at[degree, True]} | {group_sizes.at[degree]} | {short_sleep_pct.at[degree]:.2f} |")
    lines.extend([
        "",
        f"اختلاف احتمال خواب کمتر از 7 ساعت (پایین‌تر از کارشناسی منهای کارشناسی یا بالاتر) برابر {difference:.2f} واحد درصد است. این مقایسه توصیفی از پاسخ‌دهندگان نمونه، بدون وزن پیمایش و پیش از تعدیل برای سن، جنس و وضعیت تأهل است.",
    ])
    short_count = table[True].sum()
    observed = table.at[True, True]
    expected = short_count * group_sizes.at[True] / total
    gap_pp = 100 * (observed - expected) / total
    lines.extend([
        "",
        "## بررسی توصیفی استقلال خواب کمتر از 7 ساعت و تحصیلات کارشناسی یا بالاتر",
        "",
        "در صورت استقلال دو رویداد، احتمال مشترک برابر حاصل‌ضرب احتمال‌های حاشیه‌ای است. با ثابت نگه داشتن فراوانی‌های حاشیه‌ای جدول، تعداد مورد انتظار از ضرب تعداد افراد دارای A در تعداد افراد دارای B و تقسیم بر حجم نمونه به دست می‌آید.",
        "",
        "| مبنای محاسبه | تعداد افراد دارای هر دو ویژگی | درصد از کل نمونه تحلیلی |",
        "|---|---:|---:|",
        f"| مشاهده‌شده | {observed} | {100 * observed / total:.2f} |",
        f"| مورد انتظار تحت استقلال | {expected:.2f} | {100 * expected / total:.2f} |",
        "",
        f"در گروه کارشناسی یا بالاتر ({group_sizes.at[True]} نفر)، اگر فراوانی نسبی خواب کمتر از 7 ساعت برابر فراوانی آن در کل نمونه ({short_count} از {total} نفر) بود، انتظار می‌رفت {expected:.2f} نفر هر دو ویژگی را داشته باشند. اختلاف احتمال مشترک مشاهده‌شده با حاصل‌ضرب احتمال‌های حاشیه‌ای برابر {gap_pp:+.2f} واحد درصد است. این اختلاف، میزان فاصله از الگوی استقلال را در جدول بدون وزن نمونه توصیف می‌کند. استنباط درباره جامعه به بررسی عدم قطعیت با لحاظ طرح پیمایش نیاز دارد.",
    ])
    lines.extend(binomial_section(valid))
    p_degree = group_sizes.at[True] / total
    p_short = short_count / total
    p_short_given_degree = observed / group_sizes.at[True]
    p_degree_given_short = p_short_given_degree * p_degree / p_short
    if not np.isclose(p_degree_given_short, observed / short_count):
        raise ValueError("Bayes result differs from the conditional frequency.")
    lines.extend([
        "",
        "## قضیه بیز و احتمال داشتن مدرک کارشناسی یا بالاتر در گروه خواب کمتر از 7 ساعت",
        "",
        f"در نمونه دارای هر دو پاسخ معتبر، احتمال تجربی داشتن مدرک کارشناسی یا بالاتر پیش از دانستن وضعیت خواب برابر {group_sizes.at[True]} از {total} نفر ({100 * p_degree:.2f} درصد) است. برای فردی که خواب کمتر از 7 ساعت در روز کاری گزارش کرده، احتمال مورد نظر P(B|A) است.",
        "",
        r"$$P(B\mid A)=\frac{P(A\mid B)P(B)}{P(A)}.$$",
        "",
        f"در این فرمول، P(A|B) برابر {observed}/{group_sizes.at[True]}، P(B) برابر {group_sizes.at[True]}/{total} و P(A) برابر {short_count}/{total} است. با جایگذاری کسرها پیش از گردکردن، احتمال به {100 * p_degree_given_short:.2f} درصد می‌رسد.",
        "",
        f"در شمارش مستقیم نیز از {short_count} نفر با خواب کمتر از 7 ساعت، {observed} نفر مدرک کارشناسی یا بالاتر دارند. در نتیجه سهم این سطح تحصیلات در این زیرگروه {100 * (p_degree - p_degree_given_short):.2f} واحد درصد کمتر از سهم آن در کل نمونه تحلیلی است. محاسبه بدون وزن پیمایش انجام شد.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    report = build_report(project)
    output = project / "reports/phase2/sleep_education_events.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report, encoding="utf-8")
