"""Reproduce the synthetic benchmark, charts, report, and presentation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.experiments.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "experiments" / "phase6")
    parser.add_argument("--reviewed", nargs="+", type=Path, metavar="CSV")
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    if args.judgments and not args.reviewed:
        parser.error("Relevance judgments require reviewed session CSVs.")
    if args.reviewed and args.out == ROOT / "data" / "experiments" / "phase6":
        parser.error("Use a separate --out directory for reviewed results.")
    try:
        existing = args.out / "metrics.json"
        source = "reviewed" if args.reviewed else "synthetic"
        if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")).get("source") != source:
            raise ValueError("Use separate output folders for reviewed and synthetic results.")
        if args.reviewed:
            from backend.app.experiments.reviewed_evaluation import render_summary, run_reviewed_evaluation

            report = run_reviewed_evaluation(args.reviewed, args.judgments, args.seed)
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (args.out / "report.md").write_text(render_summary(report), encoding="utf-8")
            print(f"Reviewed metrics and report saved to {args.out}")
            return

        def progress(index, count):
            print(f"Evaluating participant fold {index}/{count}", flush=True)

        report = run_evaluation(args.seed, args.folds, progress)
        args.out.mkdir(parents=True, exist_ok=True)
        import matplotlib
        import pptx
        from backend.app.experiments.reporting import write_deliverables

        report["versions"].update(matplotlib=matplotlib.__version__, python_pptx=pptx.__version__)
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_deliverables(report, args.out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Synthetic evaluation, report, and presentation saved to {args.out}")


if __name__ == "__main__":
    main()
