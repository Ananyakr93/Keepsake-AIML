"""Export reviewed activity or split one or more existing CSV exports."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import dataset_format as fmt


def export_csv(args):
    from backend.app import db
    from backend.app.ml.dataset_builder import export_rows

    db.init_pool()
    try:
        user = db.get_default_user()
        rows = export_rows(user["id"], include_text=args.include_text, include_visual=args.include_visual)
    finally:
        db.close_pool()
    if not rows:
        raise ValueError("No reviewed sessions to export. Label sessions in the desktop app first.")
    args.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    contents = fmt.dataset_csv(rows)
    fd = os.open(args.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
        output.write(contents)
    print(f"Exported {len(rows)} labeled sessions to {args.path}")


def split_csv(args):
    sources = [(str(path), path.read_bytes()) for path in args.inputs]
    rows = fmt.read_datasets([(name, content.decode("utf-8-sig")) for name, content in sources])
    partitions, method = fmt.split_dataset(rows, args.seed)
    manifest = {
        "schema_version": fmt.SCHEMA_VERSION,
        "method": method,
        "seed": args.seed if method == "participant_holdout" else None,
        "input_sha256": sorted(hashlib.sha256(content).hexdigest() for _, content in sources),
        "partitions": {name: fmt.summary(entries) for name, entries in partitions.items()},
    }
    target = args.out
    if target.exists():
        raise ValueError(f"{target} already exists. Choose a new split directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dataset-", dir=target.parent) as temporary:
        staged = Path(temporary) / "split"
        staged.mkdir(mode=0o700)
        for name, entries in partitions.items():
            (staged / f"{name}.csv").write_text(fmt.dataset_csv(entries), encoding="utf-8", newline="")
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(target)
    print(f"Wrote {method} splits to {target}")
    for name, stats in manifest["partitions"].items():
        print(f"  {name}: {stats['sessions']} sessions")
        missing = [label for label, count in stats["intent_counts"].items() if count == 0]
        if missing:
            print(f"    Missing intent classes: {', '.join(missing)}")
        if 0 in stats["memory_worthy_counts"].values():
            print("    Both memory-worthiness classes are needed for binary model evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    export = commands.add_parser("export")
    export.add_argument("path", type=Path)
    export.add_argument("--include-text", action="store_true")
    export.add_argument("--include-visual", action="store_true")
    export.set_defaults(action=export_csv)
    split = commands.add_parser("split")
    split.add_argument("inputs", type=Path, nargs="+")
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(action=split_csv)
    args = parser.parse_args()
    try:
        args.action(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
