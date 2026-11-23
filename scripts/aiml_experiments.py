"""Train supervised AIML models or run the lightweight local demo."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.experiments.workspace import DemoWorkspace, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "experiments" / "synthetic")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--reviewed", nargs="+", type=Path, metavar="CSV")
    demo = commands.add_parser("demo")
    demo.add_argument("--port", type=int, default=8765)
    commands.add_parser("check-demo")

    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.reviewed and args.out == ROOT / "data" / "experiments" / "synthetic":
                parser.error("Use a separate --out directory for reviewed data.")
            report = run_training(args.out, args.reviewed or (), args.seed)
            print(json.dumps({"source": report["source"], "intent": report["intent"]["test"],
                              "admission": report["admission"]["selected"],
                              "retrieval": report.get("retrieval"), "directory": str(args.out)}, indent=2))
            return
        if not (args.out / "demo.joblib").is_file():
            run_training(args.out)
        workspace = DemoWorkspace(args.out)
        if args.command == "check-demo":
            state = workspace.summary()
            sample = workspace.analyze(state["sessions"][0]["id"])
            prompt = state["prompts"][0]["id"]
            candidates = workspace.retrieve(prompt)
            comparison = workspace.retrain(scripted=True)
            print(json.dumps({"source": "synthetic", "intent": sample["prediction"]["intent"], "candidates": len(candidates["candidates"]),
                              "top_five": sum(row["top_five"] for row in candidates["candidates"]),
                              "feedback": comparison}, indent=2))
            return
        import uvicorn
        from backend.app.experiments.web import create_demo
        uvicorn.run(create_demo(args.out), host="127.0.0.1", port=args.port)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
