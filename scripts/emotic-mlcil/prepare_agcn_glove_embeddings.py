#!/usr/bin/env python3
"""Create the audited EMOTIC 26x300 GloVe asset required by AGCN."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


GENSIM_GLOVE_300_MD5 = "29e9329ac2241937d55b852e8284e89b"


def file_hashes(path: Path):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def open_embedding_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def protocol_identity(path: Path):
    """Load normally on the server, with a stdlib-only frozen-YAML fallback."""

    try:
        from benchmarks.emotic_mlcil.protocol import load_protocol

        protocol = load_protocol(path)
        return {
            "protocol_id": protocol.protocol_id,
            "class_order": list(protocol.class_order),
            "class_order_hash": protocol.class_order_hash,
        }
    except (ModuleNotFoundError, RuntimeError):
        # This utility is intentionally usable on a local machine without the
        # PyTorch/PyYAML training environment. Only the two scalar/list fields
        # needed by the embedding asset are parsed here; the server still
        # performs the authoritative full protocol validation.
        protocol_id = None
        class_order = []
        reading_classes = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("protocol_id:"):
                protocol_id = raw_line.split(":", 1)[1].strip()
            elif raw_line == "class_order:":
                reading_classes = True
            elif raw_line == "tasks:":
                reading_classes = False
            elif reading_classes and raw_line.startswith("  - "):
                class_order.append(raw_line[4:])
        if not protocol_id or not class_order:
            raise ValueError("Could not read protocol identity from frozen YAML")
        canonical = json.dumps(
            class_order, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        return {
            "protocol_id": protocol_id,
            "class_order": class_order,
            "class_order_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }


def token_mapping(class_order):
    mapping = {}
    for name in class_order:
        if name == "Disquietment":
            # The exact EMOTIC label is absent from the fixed GloVe 6B
            # vocabulary. Freeze the closest same-root token before validation.
            mapping[name] = ["disquiet"]
        elif name == "Doubt/Confusion":
            mapping[name] = ["doubt", "confusion"]
        else:
            mapping[name] = [name.lower()]
    return mapping


def prepare(glove_path: Path, protocol_path: Path, output_path: Path):
    protocol = protocol_identity(protocol_path)
    source_sha256, source_md5 = file_hashes(glove_path)
    if glove_path.name == "glove-wiki-gigaword-300.gz" and source_md5 != GENSIM_GLOVE_300_MD5:
        raise ValueError("Gensim GloVe 300d MD5 differs from its fixed metadata")
    mapping = token_mapping(protocol["class_order"])
    required = {token for tokens in mapping.values() for token in tokens}
    found = {}
    with open_embedding_text(glove_path) as handle:
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
    for name in protocol["class_order"]:
        rows = [found[token] for token in mapping[name]]
        vectors.append([sum(values) / len(values) for values in zip(*rows)])
    payload = {
        "schema_version": 1,
        "method": "AGCN",
        "embedding_family": "GloVe 6B 300d",
        "source_name": glove_path.name,
        "source_sha256": source_sha256,
        "source_md5": source_md5,
        "source_distribution": (
            "gensim-data glove-wiki-gigaword-300; converted from Stanford "
            "GloVe text to word2vec text and gzip-compressed"
            if glove_path.suffix == ".gz"
            else "Stanford GloVe text"
        ),
        "protocol_id": protocol["protocol_id"],
        "class_order_hash": protocol["class_order_hash"],
        "class_order": protocol["class_order"],
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
        "source_md5": payload["source_md5"],
        "output_sha256": file_hashes(args.output)[0],
    }, indent=2))


if __name__ == "__main__":
    main()
