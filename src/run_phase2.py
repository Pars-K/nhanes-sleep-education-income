"""Rebuild phase 2 tables, figures, working notes and the final Persian PDF."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

from phase2_weights import build as build_weights
from sleep_education_events import build_report as events
from sleep_income import build_report as income
from sleep_marginal import build_report as marginal
from sleep_normal import build as normal


def section(text, heading):
    """Read a named Markdown section, stopping at the next heading."""
    return re.split(r"\n##? ", text.split(heading + "\n", 1)[1], maxsplit=1)[0].strip()


def markdown_html(text):
    try:
        import markdown
    except ImportError:
        import mistune
        return mistune.create_markdown(escape=False, plugins=["table"])(text)
    return markdown.markdown(text, extensions=["tables"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium", type=Path, help="Optional path to Chrome or Edge for PDF printing")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = root / "reports/phase2"
    dest.mkdir(exist_ok=True)
    for entry in (root / "data/SHA256SUMS").read_text().splitlines():
        digest, name = entry.split()
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected data version: {name}")
    parts = {"sleep_education_events": events(root), "sleep_income": income(root),
             "sleep_marginal": marginal(root / "reports/phase1")}
    for name, text in parts.items():
        (dest / f"{name}.md").write_text(text, encoding="utf-8")
    normal(root, root)
    build_weights(root)
    event = parts["sleep_education_events"]
    title = "احتمال و توزیع مدت خواب در ارتباط با تحصیلات و وضعیت اقتصادی"
    intro = f"""# {title}

**گزارش فاز دوم درس آمار | مسیر الف | NHANES اوت 2021 تا اوت 2023**

سؤال اصلی این است که در بزرگسالان 20 ساله و بالاترِ جمعیت غیرنظامی و غیرمؤسسه‌ای آمریکا، خواب معمول روزهای کاری چه ارتباطی با تحصیلات و نسبت درآمد خانواده به خط فقر دارد و پس از در نظر گرفتن سن، جنس و تأهل این ارتباط‌ها چگونه تغییر می‌کنند. در این فاز بخش احتمالی سؤال را بررسی می‌کنیم. تعدیل هم‌زمان متغیرها به مرحله رگرسیون مربوط است.

فایل خواب و فایل جمعیت‌شناختی با شناسه شرکت‌کننده به‌صورت یک‌به‌یک ادغام شدند. از 7809 بزرگسال، برای توصیف خود خواب 7698 پاسخ، برای خواب و تحصیلات 7686 جفت پاسخ و برای خواب و درآمد 6421 جفت پاسخ موجود است. مخرج هر تحلیل بر اساس متغیرهای مورد نیاز آن انتخاب شد.

## رویدادها و توزیع مشترک

""" + event.split("## احتمال‌های حاشیه‌ای", 1)[0].split("\n\n", 1)[1]
    intro = re.sub(r"واحد مشاهده یک فرد.*?\n\n", "", intro, count=1)
    intro = intro.replace("اعداد جدول فراوانی بدون وزن افراد در نمونه‌اند.", "جدول شمارش افراد بدون وزن است.")
    intro += "\n## احتمال‌های حاشیه‌ای و مشترک\n\n" + section(event, "## احتمال‌های حاشیه‌ای و مشترک")
    conditional = event[event.index("## احتمال شرطی"):event.index("## شمارش خواب")].strip()
    bayes = event[event.index("## قضیه بیز"):].strip()
    binomial = "## شمارش خواب کمتر از 7 ساعت در یک گروه 30 نفره\n\n" + section(event, "## شمارش خواب کمتر از 7 ساعت در یک گروه 30 نفره")
    weights = (dest / "sleep_weights.md").read_text(encoding="utf-8")
    shape = (dest / "sleep_normal.md").read_text(encoding="utf-8").split("برای بازتولید")[0].strip()
    moments = parts["sleep_marginal"].split("## امید ریاضی", 1)[1]
    cov = parts["sleep_income"].split("دستور بازتولید")[0].strip()
    cov = cov.replace("../../figures/phase1/weekday_sleep_by_income.png", "../../figures/phase2/sleep_income_scatter.png")
    cov = re.sub(r"نمودار فاز اول.*?دارند\.", "هر نقطه یک پاسخ جفت‌شده است. هم‌پوشانی نقاط به‌ویژه در نسبت درآمد 5 و بیشتر و ساعت‌های کامل خواب دیده می‌شود.", cov)
    cov = cov.replace("پراکندگی خواب و نسبت درآمد با نمایش تراکم مشاهدات هم‌مکان", "پراکندگی مدت خواب بر حسب نسبت درآمد خانواده به خط فقر")
    synthesis = """## جمع‌بندی و منابع

در نمونه، خواب کمتر از 7 ساعت در گروه پایین‌تر از کارشناسی شایع‌تر بود. ارتباط توصیفی نسبت درآمد با مدت خواب در جهت منفی بود. این دو نتیجه به دو ویژگی متفاوت توزیع مربوط‌اند: اولی فراوانی عبور از یک آستانه و دومی همراهی دو متغیر کمی را می‌سنجد. تفاوت ترکیب سنی و سایر ویژگی‌های گروه‌ها، اطلاعات اقتصادی ناموجود، سقف انتشار درآمد و گردکردن مدت خواب، دامنه تفسیر را محدود می‌کنند.

- [منبع داده‌های جمعیت‌شناختی و وزن مصاحبه، CDC/NCHS](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm)
- [تعریف و کدگذاری متغیرهای خواب، CDC/NCHS](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SLQ_L.htm)
- [راهنمای وزن‌دهی NHANES](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)
"""
    pages = [intro, conditional + "\n\n" + bayes, binomial, shape, "## امید ریاضی"+moments,
             weights, cov + "\n\n" + synthesis]
    (dest / "report.md").write_text("\n\n".join(p.strip() for p in pages)+"\n", encoding="utf-8")
    formulas = ["P(B | A) = P(A | B) × P(B) / P(A)", "μ = Σ xᵢ pᵢ     •     Var(X) = Σ (xᵢ − μ)² pᵢ"]
    css = (root / "src/report.css").read_text(encoding="utf-8") + """
    body{font-family:'Noto Naskh Arabic',Tahoma,'DejaVu Sans',sans-serif;font-size:10.3pt;line-height:1.7}
    article+article{break-before:page} article>*:first-child{margin-top:0}
    article img{display:block;width:100%;max-height:82mm;object-fit:contain}
    h1{font-size:17pt} h2{font-size:13pt;margin:4mm 0 2mm}
    table{font-size:9pt;margin:2mm 0 3mm} p{margin-bottom:2.4mm}
    article:first-child,article:last-child{line-height:1.57}
    article:last-child img{max-height:65mm}
    article ul{margin-top:1mm;margin-bottom:2mm}
    th,td{padding:1.1mm 1.5mm}
    .formula{direction:ltr;text-align:center;font-family:'DejaVu Sans',sans-serif}
    """
    articles = []
    for page in pages:
        formula = formulas[0] if "قضیه بیز" in page else formulas[1]
        page = re.sub(r"\$\$[^$]+\$\$", f'<div class="formula">{formula}</div>', page)
        page = re.sub(r"(?<!!)\[([^\]]+)\]\((?!https?:)[^)]+\)", r"\1", page)
        articles.append("<article>" + markdown_html(page) + "</article>")
    html = f'<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><title>{title}</title><style>{css}</style><body>'+"".join(articles)+"</body></html>"
    html_file = dest / "report.html"
    html_file.write_text(html, encoding="utf-8")
    pdf = root / "reports/phase2.pdf"
    if args.html_only:
        return
    if args.chromium:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
            subprocess.run([str(args.chromium), "--headless", "--disable-gpu", "--no-first-run",
                            "--disable-extensions", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                            f"--print-to-pdf={pdf}", html_file.as_uri()], check=True, timeout=90)
    else:
        from weasyprint import HTML
        HTML(filename=str(html_file)).write_pdf(str(pdf))
    if not pdf.is_file() or pdf.stat().st_size < 10000:
        raise RuntimeError("Missing or incomplete phase 2 PDF")


if __name__ == "__main__":
    main()
