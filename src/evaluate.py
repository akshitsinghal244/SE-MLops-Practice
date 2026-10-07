import json
import sys

from common import params


def passes_gate(accuracies, min_accuracy):
    return all(a >= min_accuracy for a in accuracies.values())


def main():
    min_acc = params()["gate"]["min_accuracy"]
    with open("metrics/ml.json") as f:
        ml = json.load(f)
    with open("metrics/dl.json") as f:
        dl = json.load(f)
    accs = {"ml": ml["accuracy"], "dl": dl["accuracy"]}
    ok = passes_gate(accs, min_acc)
    summary = {
        "ml_accuracy": accs["ml"],
        "dl_accuracy": accs["dl"],
        "best_model": max(accs, key=accs.get),
        "gate_passed": ok,
    }
    with open("metrics/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)
    if not ok:
        print(f"Quality gate FAILED (min accuracy {min_acc})")
        sys.exit(1)


if __name__ == "__main__":
    main()
