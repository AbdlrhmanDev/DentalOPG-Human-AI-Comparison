
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
path = root / "data/derived/per_image_detection_counts.csv"

df = pd.read_csv(path)

rows = []

for (evaluator, iou), g in df.groupby(
    ["participant", "threshold"]
):
    tp = int(g["TP"].sum())
    fp = int(g["FP"].sum())
    fn = int(g["FN"].sum())

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn)

    rows.append({
        "Evaluator": evaluator,
        "IoU": iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(p, 6),
        "Recall": round(r, 6),
        "Micro-F1": round(f1, 6)
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
