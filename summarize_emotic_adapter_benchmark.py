import csv
import json
from html import escape
from pathlib import Path


METHODS = ("ddp", "internal", "external", "hybrid")


def main():
    output_dir = Path("output/emotic_adapter_inference_benchmark")
    rows = []
    for method in METHODS:
        path = output_dir / f"{method}.json"
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        rows.append(
            {
                key: data[key]
                for key in (
                    "method",
                    "batch_size",
                    "latency_ms_mean",
                    "latency_ms_std",
                    "peak_memory_mib",
                    "peak_reserved_memory_mib",
                    "loaded_parameters",
                    "adapter_parameters_loaded",
                    "extra_image_encoder",
                    "device",
                )
            }
        )
    with open(output_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "benchmark.json", "w", encoding="utf-8") as fp:
        json.dump({"rows": rows}, fp, indent=2, ensure_ascii=False)
    headers = "".join(f"<th>{escape(key)}</th>" for key in rows[0])
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in row)
        + "</tr>"
        for row in rows
    )
    (output_dir / "benchmark.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Adapter Benchmark</title>"
        "<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:right}</style>"
        f"<h1>EMOTIC Adapter Inference Benchmark</h1><table><tr>{headers}</tr>"
        f"{body}</table>",
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"{row['method']:10s} latency={row['latency_ms_mean']:.2f} ms "
            f"peak={row['peak_memory_mib']:.1f} MiB"
        )


if __name__ == "__main__":
    main()
