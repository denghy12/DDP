#!/usr/bin/env python3
"""Create the audited EMOTIC 26x300 GloVe asset required by AGCN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.emotic_mlcil.protocol import load_protocol


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_mapping(class_order):
    mapping = {}
    for name in class_order:
        if name == "Doubt/Confusion":
            mapping[name] = ["doubt", "confusion"]
        else:
            mapping[name] = [name.lower()]
    return mapping


def prepare(glove_path: Path, protocol_path: Path, output_path: Path):
    protocol = load_protocol(protocol_path)
    mapping = token_mapping(protocol.class_order)
    required = {token for tokens in mapping.values() for token in tokens}
    found = {}
    with glove_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split(" ")
            if fields[0] not in required:
                continue
            vector = [float(item) for item in fields[1:]]
            if len(vector) != 300:
                raise ValueError(f"GloVe token {fields[0]!r} is not 300-dimensional")
            found[fields[0]] = vector
            if len(found) == len(required):
                break
    missing = sorted(required.difference(found))
    if missing:
        raise ValueError("Missing required GloVe tokens: " + ", ".join(missing))
    vectors = []
    for name in protocol.class_order:
        rows = [found[token] for token in mapping[name]]
        vectors.append([sum(values) / len(values) for values in zip(*rows)])
    payload = {
        "schema_version": 1,
        "method": "AGCN",
        "embedding_family": "GloVe 6B 300d",
        "source_name": glove_path.name,
        "source_sha256": file_sha256(glove_path),
        "protocol_id": protocol.protocol_id,
        "class_order_hash": protocol.class_order_hash,
        "class_order": list(protocol.class_order),
        "token_mapping": mapping,
        "vectors": vectors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", required=True, type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/emotic_mlcil/protocol_b5c3.yaml")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("pretrained/agcn/emotic_glove_6b_300d.json"),
    )
    args = parser.parse_args()
    payload = prepare(args.glove, args.protocol, args.output)
    print(json.dumps({
        "output": str(args.output),
        "classes": len(payload["vectors"]),
        "dimensions": len(payload["vectors"][0]),
        "source_sha256": payload["source_sha256"],
        "output_sha256": file_sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
