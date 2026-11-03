"""بررسی مستقل جدول‌های عددی، خروجی گزارش و فایل‌های مورد نیاز فاز اول."""

import hashlib
import math
from pathlib import Path
import re
import statistics

import pandas as pd
from pypdf import PdfReader
from pypdf.generic import ContentStream

from describe_categories import CATEGORIES


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/phase1"


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-10):
        raise AssertionError(f"اختلاف عددی: {actual} != {expected}")


def mean_reference(data, column):
    pairs = list(data[[column, "WTINT2YR"]].itertuples(index=False, name=None))
    return (
        math.fsum(x for x, _ in pairs) / len(pairs),
        math.fsum(x * w for x, w in pairs) / math.fsum(w for _, w in pairs),
    )


def main():
    for line in (ROOT / "data/SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        if hashlib.sha256((ROOT / filename.strip()).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"هش فایل خام تغییر کرده است: {filename}")
    demo = pd.read_sas(ROOT / "data/raw/DEMO_L.xpt", format="xport", encoding=None)
    sleep = pd.read_sas(ROOT / "data/raw/SLQ_L.xpt", format="xport", encoding=None)
    assert demo.SEQN.notna().all() and demo.SEQN.is_unique
    assert sleep.SEQN.notna().all() and sleep.SEQN.is_unique
    assert set(sleep.SEQN).issubset(set(demo.SEQN))
    adults = demo.merge(sleep, how="left", on="SEQN", validate="one_to_one").loc[lambda x: x.RIDAGEYR.ge(20)]
    assert len(demo) == 11933 and len(sleep) == 8501 and len(adults) == 7809
    numeric = pd.read_csv(REPORTS / "numeric_summaries.csv")
    weighted_numeric = pd.read_csv(REPORTS / "weighted_numeric_comparisons.csv")
    for _, row in numeric.iterrows():
        column = row["نام متغیر"]
        data = adults.loc[adults[column].notna()]
        values = data[column].tolist()
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        assert row["تعداد"] == len(values)
        assert row["تعداد ناموجود"] + len(values) == len(adults)
        for name, expected in {
            "میانگین": statistics.mean(values), "میانه": statistics.median(values),
            "انحراف معیار": statistics.stdev(values), "چارک اول": q1,
            "چارک سوم": q3, "دامنه میان‌چارکی": q3-q1,
        }.items():
            close(row[name], expected)
        simple, weighted = mean_reference(data, column)
        saved = weighted_numeric.loc[weighted_numeric["نام متغیر"].eq(column)].iloc[0]
        close(saved["میانگین بدون وزن"], simple)
        close(saved["میانگین وزندار"], weighted)

    categories = pd.read_csv(REPORTS / "weighted_categorical_comparisons.csv")
    groups = pd.read_csv(REPORTS / "weighted_sleep_by_group.csv")
    sensitivity = pd.read_csv(REPORTS / "weighted_group_sleep_sensitivity.csv")
    for column, (_, labels, _) in CATEGORIES.items():
        valid = adults.loc[adults[column].isin(labels)]
        for code, label in labels.items():
            group = valid.loc[valid[column].eq(code)]
            saved = categories.loc[categories["نام متغیر"].eq(column) & categories["گروه"].eq(label)].iloc[0]
            assert saved["فراوانی نمونه"] == len(group)
            close(saved["درصد بدون وزن"], 100 * len(group) / len(valid))
            close(saved["درصد وزندار"], 100 * math.fsum(group.WTINT2YR) / math.fsum(valid.WTINT2YR))
            with_sleep = group.loc[group.SLD012.notna()]
            u, w = mean_reference(with_sleep, "SLD012")
            saved = groups.loc[groups["نام متغیر"].eq(column) & groups["گروه"].eq(label)].iloc[0]
            assert saved["تعداد پاسخ مشترک"] == len(with_sleep)
            close(saved["میانگین خواب بدون وزن؛ ساعت"], u)
            close(saved["میانگین خواب وزندار؛ ساعت"], w)
            inner = with_sleep.loc[~with_sleep.SLD012.isin([2, 14])]
            iu, iw = mean_reference(inner, "SLD012")
            saved = sensitivity.loc[sensitivity["نام متغیر"].eq(column) & sensitivity["گروه"].eq(label)].iloc[0]
            close(saved["میانگین بدون وزن پس از کنارگذاشتن؛ ساعت"], iu)
            close(saved["میانگین وزندار پس از کنارگذاشتن؛ ساعت"], iw)
            close(saved["تغییر میانگین بدون وزن؛ دقیقه"], 60 * (iu-u))
            close(saved["تغییر میانگین وزندار؛ دقیقه"], 60 * (iw-w))

    distributions = pd.read_csv(REPORTS / "weekday_distribution_weight_comparison.csv")
    with_sleep = adults.loc[adults.SLD012.notna()]
    for _, row in distributions.iterrows():
        group = with_sleep.loc[with_sleep.SLD012.eq(row["کد منتشرشده"])]
        close(row["درصد بدون وزن"], 100 * len(group) / len(with_sleep))
        close(row["درصد وزندار"], 100 * math.fsum(group.WTINT2YR) / math.fsum(with_sleep.WTINT2YR))
    close(distributions["درصد بدون وزن"].sum(), 100)
    close(distributions["درصد وزندار"].sum(), 100)

    for path in [*ROOT.glob("*.md"), *REPORTS.glob("*.md")]:
        text = path.read_text()
        assert text.count("```") % 2 == 0, path
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("https:", "http:")):
                assert (path.parent / target).exists(), (path, target)
    non_english_digits = re.compile("[\u06f0-\u06f9\u0660-\u0669]")
    for path in (ROOT / "reports").rglob("*"):
        if path.suffix in (".md", ".html", ".csv"):
            text = path.read_text(encoding="utf-8-sig")
            assert not non_english_digits.search(text), path
            assert "PR.pdf" not in text, path
    for path in [ROOT / "reports/phase1.pdf", *(ROOT / "figures").rglob("*.pdf")]:
        reader = PdfReader(path)
        assert "/CreationDate" not in reader.metadata and "/ModDate" not in reader.metadata, path
        for page in reader.pages:
            assert not non_english_digits.search(page.extract_text()), path
    reader = PdfReader(ROOT / "reports/phase1.pdf")
    image_count = 0
    for page in reader.pages:
        assert page.extract_text().strip()
        for operands, operator in ContentStream(page.get_contents(), reader).operations:
            if operator == b"Do" and page["/Resources"]["/XObject"][operands[0]].get("/Subtype") == "/Image":
                image_count += 1
    assert image_count == 7, image_count
    assert len(PdfReader(ROOT / "figures/phase1/all_seven_plots.pdf").pages) == 7
    print("بررسی مستقل شاخص‌ها، وزن‌ها، تحلیل حساسیت، پیوندها، رقم‌های انگلیسی و هفت نمودار انجام شد.")


if __name__ == "__main__":
    main()
