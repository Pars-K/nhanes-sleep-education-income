"""اجرای کامل فاز اول از فایل‌های خام تا گزارش PDF، بدون عملیات Git."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "inspect_data.py",
    "audit_adults.py",
    "describe_categories.py",
    "describe_numeric.py",
    "compare_weights.py",
    "compare_all_weights.py",
    "plot_sleep_relationships.py",
    "build_phase1_report.py",
    "validate_phase1.py",
]


def main():
    for filename in ("DEMO_L.xpt", "SLQ_L.xpt"):
        if not (ROOT / "data/raw" / filename).is_file():
            raise FileNotFoundError(f"فایل خام موجود نیست؛ راهنمای دریافت داده‌ها را ببینید: {filename}")
    for script in SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / script)], cwd=ROOT,
            capture_output=True, text=True,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"انجام شد: {script}", flush=True)


if __name__ == "__main__":
    main()
