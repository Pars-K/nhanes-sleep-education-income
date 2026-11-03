"""بررسی ساختار فایل‌های خام؛ بدون ادغام، تغییر یا حذف مشاهدات."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "DEMO_L": [
        "SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", "DMDMARTZ",
        "INDFMPIR", "WTINT2YR", "SDMVSTRA", "SDMVPSU",
    ],
    "SLQ_L": ["SEQN", "SLD012", "SLD013"],
}


def main():
    tables = {}
    summary = {}
    for name, expected in EXPECTED.items():
        table = pd.read_sas(
            ROOT / "data" / "raw" / f"{name}.xpt", format="xport", encoding=None
        )
        absent = sorted(set(expected) - set(table.columns))
        if absent:
            raise ValueError(f"ستون‌های مورد انتظار یافت نشدند: {name}: {absent}")
        tables[name] = table
        summary[name] = {
            "تعداد ردیف": len(table),
            "تعداد ستون": len(table.columns),
            "تعداد شناسه یکتا": int(table["SEQN"].nunique()),
            "تعداد شناسه گمشده": int(table["SEQN"].isna().sum()),
            "تعداد تکرار شناسه پس از اولین مشاهده": int(table["SEQN"].duplicated().sum()),
            "ستون‌ها": list(table.columns),
        }

    demo_ids = set(tables["DEMO_L"]["SEQN"].dropna())
    sleep_ids = set(tables["SLQ_L"]["SEQN"].dropna())
    summary["اشتراک شناسه‌ها"] = {
        "مشترک": len(demo_ids & sleep_ids),
        "فقط جمعیت‌شناسی": len(demo_ids - sleep_ids),
        "فقط خواب": len(sleep_ids - demo_ids),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
