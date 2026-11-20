"""Empirical covariance of weekday sleep and the family income-to-poverty ratio."""

from pathlib import Path

import numpy as np
import pandas as pd


def build_report(root):
    raw = root / "data/raw"
    demo = pd.read_sas(raw / "DEMO_L.xpt", format="xport")
    sleep = pd.read_sas(raw / "SLQ_L.xpt", format="xport")
    adults = demo.loc[demo.RIDAGEYR.ge(20), ["SEQN", "INDFMPIR", "WTINT2YR"]]
    pairs = adults.merge(sleep[["SEQN", "SLD012"]], how="left", on="SEQN",
                         validate="one_to_one").dropna(subset=["INDFMPIR", "SLD012"])
    x, z, w = pairs.SLD012, pairs.INDFMPIR, pairs.WTINT2YR
    if not np.isfinite(w).all() or not w.gt(0).all():
        raise ValueError("Missing or invalid interview weights in the paired sample.")
    counts = pd.read_csv(root / "reports/phase1/relationship_plot_counts.csv")
    plot_n = counts.loc[counts["متغیر توضیحی"].eq("نسبت درآمد خانواده به خط فقر"),
                        "تعداد واردشده در نمودار"].item()
    if len(pairs) != plot_n:
        raise ValueError("Paired sample differs from phase 1 income plot.")

    lines = [
        "# کوواریانس مدت خواب و نسبت درآمد خانواده به خط فقر",
        "",
        "برای هر فرد، X مدت خواب روز کاری بر حسب ساعت و Z نسبت درآمد خانواده به خط فقر است. از مقادیر منتشرشده استفاده کردیم. کدهای 2 و 14 خواب تقریب دو رده انتهایی‌اند و کد 5 درآمد نماینده نسبت 5 و بالاتر است.",
        "",
        f"از {len(adults)} بزرگسال، {len(pairs)} نفر هر دو پاسخ را داشتند. در هر محاسبه، میانگین هر دو متغیر و کوواریانس روی همین افراد به دست آمد. {z.eq(5).sum()} نفر از این گروه در رده بالایی درآمد قرار دارند.",
        "",
        "![پراکندگی خواب و نسبت درآمد با نمایش تراکم مشاهدات هم‌مکان](../../figures/phase1/weekday_sleep_by_income.png)",
        "",
        "نمودار فاز اول پراکندگی همین زوج‌ها را با خانه‌های شش‌ضلعی نشان می‌دهد. رنگ هر خانه تعداد مشاهدات بدون وزن در آن خانه است. تجمع در سمت راست نمودار با سقف انتشار درآمد و ردیف‌های افقی با ثبت نیم‌ساعته خواب ارتباط دارند.",
        "",
        "برای هر فرد، انحراف خواب از میانگین خواب را در انحراف نسبت درآمد از میانگین نسبت درآمد ضرب کردیم. میانگین این حاصل‌ضرب‌ها با تقسیم بر تعداد افراد، کوواریانس تجربی را می‌دهد. در نسخه وزندار، هر دو میانگین و میانگین حاصل‌ضرب با وزن مصاحبه و مخرج مجموع وزن‌ها محاسبه شدند.",
        "",
        "| توزیع تجربی | میانگین خواب (ساعت) | میانگین نسبت درآمد | کوواریانس (ساعت × واحد نسبت درآمد) |",
        "|---|---:|---:|---:|",
    ]
    for label, weights in [("بدون وزن", None), ("با وزن مصاحبه", w)]:
        mean_x = np.average(x, weights=weights)
        mean_z = np.average(z, weights=weights)
        covariance = np.average((x - mean_x) * (z - mean_z), weights=weights)
        lines.append(f"| {label} | {mean_x:.4f} | {mean_z:.4f} | {covariance:.4f} |")
    lines.extend([
        "",
        "علامت منفی در هر دو محاسبه نشان می‌دهد که در مجموع، نسبت درآمد بالاتر با خواب ثبت‌شده کوتاه‌تر همراه بوده است. کوواریانس به مقیاس متغیرها وابسته است. برای برآورد تغییر مدت خواب به ازای یک واحد افزایش نسبت درآمد، باید شیب رگرسیون را محاسبه کرد. این مقایسه توصیفی پیش از تعدیل سن، جنس و وضعیت تأهل است.",
        "",
        "دستور بازتولید از ریشه پروژه: `python src/sleep_income.py`.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    output = project / "reports/phase2/sleep_income.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(build_report(project), encoding="utf-8")
