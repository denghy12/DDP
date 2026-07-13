#!/usr/bin/env python3
"""Export external Prototype fusion reports in the same HTML format as DDP.

The local Python on some machines may not have torch installed.  The fusion
score files are simple torch.save zip archives containing FloatStorage tensors,
so this script reads the needed tensors with a tiny stdlib-only loader and then
emits the exact DetailReport-style table used by
src/helper_functions/detail_report.py.
"""

import array
import collections
import io
import json
import os
import pickle
import statistics
import sys
import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path


ROOT = Path("output")
SUMMARY_DIR = ROOT / "emotic_prototype_fusion_strict_summary"
DETAIL_DIR = SUMMARY_DIR / "detail"

DETAIL_METRICS = [
    "ap",
    "f1",
    "precision",
    "recall",
    "support",
    "predicted_positive",
    "tp",
    "fp",
    "fn",
    "tn",
]

CONDITIONS = (
    {
        "run_name": "external_16shot",
        "title": "External 16-shot Prototype Fusion",
        "dirs": [
            ROOT / f"emotic_prototype_fusion_strict_base5_16shot_seed{seed}"
            for seed in range(3)
        ],
    },
    {
        "run_name": "external_full",
        "title": "External Full Base5 Prototype Fusion",
        "dirs": [
            ROOT / f"emotic_prototype_fusion_strict_full_seed{seed}"
            for seed in range(3)
        ],
    },
)


CLASS_MASK = [
    list(range(0, 5)),
    list(range(5, 8)),
    list(range(8, 11)),
    list(range(11, 14)),
    list(range(14, 17)),
    list(range(17, 20)),
    list(range(20, 23)),
    list(range(23, 26)),
]


@dataclass(frozen=True)
class TensorRef:
    storage_key: str
    storage_offset: int
    size: tuple
    stride: tuple


class TorchArchiveUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        storage_kind, _storage_type, storage_key, _location, numel = pid
        if storage_kind != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
        return {"storage_key": str(storage_key), "numel": int(numel)}

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor
        if module == "torch" and name == "FloatStorage":
            return "FloatStorage"
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return super().find_class(module, name)


def rebuild_tensor(
    storage,
    storage_offset,
    size,
    stride,
    _requires_grad,
    _backward_hooks,
):
    return TensorRef(
        storage_key=storage["storage_key"],
        storage_offset=int(storage_offset),
        size=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
    )


def torch_load_float_archive(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        prefix = names[0].split("/", 1)[0]
        payload = TorchArchiveUnpickler(
            io.BytesIO(archive.read(f"{prefix}/data.pkl"))
        ).load()

        storages = {}
        for name in names:
            marker = f"{prefix}/data/"
            if not name.startswith(marker):
                continue
            key = name[len(marker) :]
            values = array.array("f")
            values.frombytes(archive.read(name))
            if sys.byteorder != "little":
                values.byteswap()
            storages[key] = values

    def materialize(value):
        if not isinstance(value, TensorRef):
            return value
        storage = storages[value.storage_key]
        if len(value.size) == 1:
            return [
                float(storage[value.storage_offset + i * value.stride[0]])
                for i in range(value.size[0])
            ]
        if len(value.size) == 2:
            rows, cols = value.size
            row_stride, col_stride = value.stride
            return [
                [
                    float(
                        storage[
                            value.storage_offset
                            + row * row_stride
                            + col * col_stride
                        ]
                    )
                    for col in range(cols)
                ]
                for row in range(rows)
            ]
        raise ValueError(f"Unsupported tensor size: {value.size}")

    return {key: materialize(value) for key, value in payload.items()}


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def average_precision(scores, targets):
    positives = int(sum(1 for value in targets if value > 0.5))
    if positives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if targets[index] > 0.5:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def binary_counts(predictions, targets):
    tp = fp = fn = tn = 0
    for prediction, target in zip(predictions, targets):
        pred = bool(prediction)
        truth = target > 0.5
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    predicted_positive = tp + fp
    support = tp + fn
    precision = safe_divide(tp, predicted_positive)
    recall = safe_divide(tp, support)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "support": support,
        "predicted_positive": predicted_positive,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_class_metrics(scores, targets, class_id, threshold):
    class_scores = [row[class_id] for row in scores]
    class_targets = [row[class_id] for row in targets]
    predictions = [score > threshold for score in class_scores]
    metrics = binary_counts(predictions, class_targets)
    metrics["ap"] = average_precision(class_scores, class_targets)
    return metrics


def mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return statistics.mean(values)


def seen_classes_for_task(task_id):
    seen = []
    for task_classes in CLASS_MASK[: task_id + 1]:
        seen.extend(task_classes)
    return seen


def class_task_map():
    output = {}
    for task_id, classes in enumerate(CLASS_MASK):
        for class_id in classes:
            output[class_id] = task_id
    return output


def load_run(directory):
    summary_path = directory / "fusion_all_tasks_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open() as handle:
        summary = json.load(handle)
    return summary


def load_task_scores(directory, task_id):
    path = directory / f"task{task_id}_fusion_scores.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch_load_float_archive(path)
    val_count = int(payload["val_count"])
    return {
        "scores": payload["binary_gate_scores"][val_count:],
        "targets": payload["targets"][val_count:],
        "threshold": float(payload["thresholds"]["binary_gate"]),
    }


def aggregate_condition(condition):
    runs = []
    for seed, directory in enumerate(condition["dirs"]):
        runs.append(
            {
                "seed": seed,
                "directory": directory,
                "summary": load_run(directory),
            }
        )

    class_names = runs[0]["summary"]["tasks"][-1]["classnames"]
    task_map = class_task_map()
    rows = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "class_task": task_map[class_id],
            "count": None,
        }
        for class_id, class_name in enumerate(class_names)
    ]
    overall_rows = []
    per_seed_overall = []

    for task_id in range(len(CLASS_MASK)):
        seen_classes = seen_classes_for_task(task_id)
        per_seed_task_metrics = []
        per_seed_task_overall = []

        for run in runs:
            score_payload = load_task_scores(run["directory"], task_id)
            scores = score_payload["scores"]
            targets = score_payload["targets"]
            threshold = score_payload["threshold"]
            task_metrics = {}
            for class_id in seen_classes:
                task_metrics[class_id] = compute_class_metrics(
                    scores, targets, class_id, threshold
                )
            per_seed_task_metrics.append(task_metrics)

            summary_task = run["summary"]["tasks"][task_id]
            overall = summary_task["results"]["binary_gate"]["test"]
            per_seed_task_overall.append(
                {
                    "mAP": float(overall["mAP"]),
                    "oF1": float(overall["oF1"]),
                    "cF1": float(overall["cF1"]),
                    "samples": int(overall["samples"]),
                }
            )

        for class_id in seen_classes:
            for metric in DETAIL_METRICS:
                rows[class_id][f"task{task_id}_{metric}"] = mean(
                    [
                        task_metrics[class_id][metric]
                        for task_metrics in per_seed_task_metrics
                    ]
                )
            if rows[class_id]["count"] is None:
                rows[class_id]["count"] = int(
                    round(rows[class_id][f"task{task_id}_support"])
                )

        current = {
            "task": task_id,
            "seen_classes": len(seen_classes),
            "samples": int(round(mean([row["samples"] for row in per_seed_task_overall]))),
            "mAP": mean([row["mAP"] for row in per_seed_task_overall]),
            "oF1": mean([row["oF1"] for row in per_seed_task_overall]),
            "cF1": mean([row["cF1"] for row in per_seed_task_overall]),
            "loss": None,
        }
        per_seed_overall.append(current)
        current["amAP"] = mean([row["mAP"] for row in per_seed_overall])
        overall_rows.append(current)

    return class_names, rows, overall_rows


def write_detail_report(path, title, table_columns, table_rows, overall_columns, overall_rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    escaped_title = escape(title)
    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin: 0 0 14px; font-size: 28px; line-height: 1.2; }}
    h2 {{ margin: 18px 0 10px; font-size: 18px; line-height: 1.2; }}
    .toolbar {{ align-items: center; display: flex; gap: 12px; margin-bottom: 12px; }}
    .segmented-control {{ display: inline-flex; border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; background: #ffffff; }}
    .segmented-control button {{ border: 0; border-right: 1px solid #cbd5e1; background: #ffffff; color: #334155; cursor: pointer; font-size: 13px; padding: 7px 12px; }}
    .segmented-control button:last-child {{ border-right: 0; }}
    .segmented-control button.active {{ background: #0f172a; color: #ffffff; }}
    .layout-hint {{ color: #64748b; font-size: 13px; }}
    .table-wrap {{ max-height: calc(100vh - 130px); overflow: auto; border: 1px solid #d1d5db; }}
    .overall-wrap {{ max-height: 240px; overflow: auto; border: 1px solid #d1d5db; margin-bottom: 16px; }}
    table.metric-table {{ border-collapse: collapse; font-size: 12px; white-space: nowrap; }}
    .metric-table th, .metric-table td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
    .metric-table th {{ position: sticky; top: 0; background: #f3f4f6; z-index: 2; }}
    .metric-table th:nth-child(1), .metric-table td:nth-child(1) {{ position: sticky; left: 0; min-width: 64px; text-align: left; background: #ffffff; z-index: 1; }}
    .metric-table th:nth-child(2), .metric-table td:nth-child(2) {{ position: sticky; left: 64px; min-width: 150px; text-align: left; background: #ffffff; z-index: 1; }}
    .metric-table th:nth-child(3), .metric-table td:nth-child(3) {{ position: sticky; left: 214px; min-width: 80px; text-align: left; background: #ffffff; z-index: 1; }}
    .metric-table th:nth-child(4), .metric-table td:nth-child(4) {{ position: sticky; left: 294px; min-width: 80px; text-align: right; background: #ffffff; z-index: 1; box-shadow: 1px 0 0 #e5e7eb; }}
    .metric-table th:nth-child(-n+4) {{ background: #f3f4f6; z-index: 3; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <h2>总体指标</h2>
  <div class="overall-wrap"><table class="metric-table" id="overallTable"></table></div>
  <div class="toolbar">
    <div class="segmented-control" role="group" aria-label="column layout">
      <button type="button" class="active" data-layout="task">按 task 分组</button>
      <button type="button" data-layout="metric">按指标分组</button>
    </div>
    <span class="layout-hint" id="layoutHint"></span>
  </div>
  <div class="table-wrap"><table class="metric-table" id="metricTable"></table></div>
  <script>
    const tableColumns = {json.dumps(table_columns, ensure_ascii=False)};
    const tableRows = {json.dumps(table_rows, ensure_ascii=False, allow_nan=False)};
    const overallColumns = {json.dumps(overall_columns, ensure_ascii=False)};
    const overallRows = {json.dumps(overall_rows, ensure_ascii=False, allow_nan=False)};
    const metricPreference = {json.dumps(DETAIL_METRICS, ensure_ascii=False)};
    const frozenColumnCount = 4;
    const indexColumns = tableColumns.slice(0, frozenColumnCount);
    const metricColumns = tableColumns.slice(frozenColumnCount);
    function parseMetricColumn(column) {{
      const match = /^task(\\d+)_(.+)$/.exec(column);
      return match ? {{column, task: Number(match[1]), metric: match[2]}} : null;
    }}
    const parsedColumns = metricColumns.map(parseMetricColumn).filter(Boolean);
    const tasks = [...new Set(parsedColumns.map((item) => item.task))].sort((a,b) => a-b);
    const discoveredMetrics = [...new Set(parsedColumns.map((item) => item.metric))];
    const metrics = [...metricPreference.filter((m) => discoveredMetrics.includes(m)), ...discoveredMetrics.filter((m) => !metricPreference.includes(m))];
    function taskFirstColumns() {{
      return [...indexColumns, ...tasks.flatMap((task) => metrics.map((metric) => `task${{task}}_${{metric}}`).filter((column) => tableColumns.includes(column)))];
    }}
    function metricFirstColumns() {{
      return [...indexColumns, ...metrics.flatMap((metric) => tasks.map((task) => `task${{task}}_${{metric}}`).filter((column) => tableColumns.includes(column)))];
    }}
    function formatValue(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "NaN";
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6);
      return String(value);
    }}
    function renderStaticTable(tableId, columns, rows) {{
      const table = document.getElementById(tableId);
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      columns.forEach((column) => {{ const th=document.createElement("th"); th.textContent=column; headerRow.appendChild(th); }});
      thead.appendChild(headerRow);
      const tbody = document.createElement("tbody");
      rows.forEach((row) => {{
        const tr=document.createElement("tr");
        columns.forEach((column) => {{ const td=document.createElement("td"); td.textContent=formatValue(row[column]); tr.appendChild(td); }});
        tbody.appendChild(tr);
      }});
      table.replaceChildren(thead, tbody);
    }}
    function setLayout(layout) {{
      renderStaticTable("metricTable", layout === "metric" ? metricFirstColumns() : taskFirstColumns(), tableRows);
      document.querySelectorAll("[data-layout]").forEach((button) => button.classList.toggle("active", button.dataset.layout === layout));
      document.getElementById("layoutHint").textContent = layout === "metric"
        ? "当前列顺序：task0-7 的 ap，然后 task0-7 的 f1，依次类推。"
        : "当前列顺序：task0 的所有指标，然后 task1 的所有指标，依次类推。";
    }}
    renderStaticTable("overallTable", overallColumns, overallRows);
    document.querySelectorAll("[data-layout]").forEach((button) => button.addEventListener("click", () => setLayout(button.dataset.layout)));
    setLayout("task");
  </script>
</body>
</html>
'''
    Path(path).write_text(html, encoding="utf-8")


def render_condition(condition):
    _class_names, rows, overall_rows = aggregate_condition(condition)
    table_columns = ["class_id", "class_name", "class_task", "count"]
    for task_id in range(len(CLASS_MASK)):
        for metric in DETAIL_METRICS:
            table_columns.append(f"task{task_id}_{metric}")
    table_rows = [{column: row.get(column) for column in table_columns} for row in rows]
    overall_columns = [
        "task",
        "seen_classes",
        "samples",
        "mAP",
        "amAP",
        "oF1",
        "cF1",
        "loss",
    ]
    overall_rows = [
        {column: row.get(column) for column in overall_columns}
        for row in overall_rows
    ]
    title = f"{condition['run_name']}_per_class_metrics"
    canonical_path = (
        DETAIL_DIR / f"{condition['run_name']}_per_class_task_table.html"
    )
    write_detail_report(
        canonical_path,
        title,
        table_columns,
        table_rows,
        overall_columns,
        overall_rows,
    )

    # Keep the previously shared filenames valid, but with the corrected
    # DetailReport-compatible HTML format.
    legacy_path = SUMMARY_DIR / f"{condition['run_name']}_per_class_detail.html"
    legacy_path.write_text(canonical_path.read_text(encoding="utf-8"), encoding="utf-8")
    return canonical_path, legacy_path


def main():
    written = []
    for condition in CONDITIONS:
        written.extend(render_condition(condition))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
