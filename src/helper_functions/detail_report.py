import json
import os
from html import escape

import numpy as np
import torch


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


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def average_precision(scores, targets):
    positives = int(targets.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].float()
    true_positive_cumsum = torch.cumsum(sorted_targets, dim=0)
    rank = torch.arange(1, sorted_targets.numel() + 1).float()
    precision_at_rank = true_positive_cumsum / rank
    return float((precision_at_rank * sorted_targets).sum().item() / positives)


def binary_counts(predictions, targets):
    predictions = predictions.bool()
    targets = targets.bool()
    tp = int((predictions & targets).sum().item())
    fp = int((predictions & ~targets).sum().item())
    fn = int((~predictions & targets).sum().item())
    tn = int((~predictions & ~targets).sum().item())
    predicted_positive = int(predictions.sum().item())
    support = int(targets.sum().item())
    precision = _safe_divide(tp, predicted_positive)
    recall = _safe_divide(tp, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
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


def compute_multilabel_metrics(scores, targets, threshold):
    scores = scores.detach().cpu()
    targets = targets.detach().cpu().bool()
    per_class = []
    for class_id in range(scores.shape[1]):
        counts = binary_counts(scores[:, class_id].gt(threshold), targets[:, class_id])
        counts["ap"] = average_precision(
            scores[:, class_id], targets[:, class_id].float()
        )
        per_class.append(counts)

    map_value = 100.0 * float(np.mean([item["ap"] for item in per_class]))
    overall = binary_counts(scores.gt(threshold), targets)
    cf1 = 100.0 * float(np.mean([item["f1"] for item in per_class]))
    of1 = 100.0 * overall["f1"]
    return map_value, cf1, of1, per_class


class DetailReport:
    def __init__(self, output_dir, run_name, class_names, class_mask, val_targets):
        self.output_dir = output_dir
        self.run_name = run_name
        self.class_names = list(class_names)
        self.class_mask = [list(task) for task in class_mask]
        self.rows = []
        self.overall_rows = []

        class_tasks = {}
        for task_id, classes in enumerate(self.class_mask):
            for class_id in classes:
                class_tasks[int(class_id)] = task_id
        counts = [0] * len(self.class_names)
        for target in val_targets:
            for class_id in target:
                counts[int(class_id)] += 1

        for class_id, class_name in enumerate(self.class_names):
            self.rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "class_task": class_tasks[class_id],
                    "count": counts[class_id],
                }
            )

    @property
    def path(self):
        return os.path.join(
            self.output_dir,
            "detail",
            f"{self.run_name}_per_class_task_table.html",
        )

    def update(self, task_id, scores, targets, threshold, loss):
        seen_classes = sorted(
            class_id
            for task in self.class_mask[: task_id + 1]
            for class_id in task
        )
        clean_scores = scores[:, seen_classes]
        clean_targets = targets[:, seen_classes]
        map_value, cf1, of1, per_class = compute_multilabel_metrics(
            clean_scores, clean_targets, threshold
        )

        for local_id, class_id in enumerate(seen_classes):
            for metric in DETAIL_METRICS:
                self.rows[class_id][f"task{task_id}_{metric}"] = per_class[local_id][
                    metric
                ]

        self.overall_rows.append(
            {
                "task": task_id,
                "seen_classes": len(seen_classes),
                "samples": int(targets.shape[0]),
                "mAP": map_value,
                "amAP": float(
                    np.mean(
                        [row["mAP"] for row in self.overall_rows] + [map_value]
                    )
                ),
                "oF1": of1,
                "cF1": cf1,
                "loss": float(loss),
            }
        )
        self.write()
        return self.overall_rows[-1]

    def write(self):
        table_columns = ["class_id", "class_name", "class_task", "count"]
        for task_id in range(len(self.class_mask)):
            for metric in DETAIL_METRICS:
                table_columns.append(f"task{task_id}_{metric}")

        table_rows = []
        for row in self.rows:
            table_rows.append({column: row.get(column) for column in table_columns})
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
            for row in self.overall_rows
        ]
        _write_detail_report(
            self.path,
            f"{self.run_name}_per_class_metrics",
            table_columns,
            table_rows,
            overall_columns,
            overall_rows,
        )


def _write_detail_report(
    path, title, table_columns, table_rows, overall_columns, overall_rows
):
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
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(html)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "table_columns": table_columns,
                "table_rows": table_rows,
                "overall_columns": overall_columns,
                "overall_rows": overall_rows,
            },
            fp,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

