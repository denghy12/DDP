#!/usr/bin/env python3
"""Conservative AGCN visual/GNN/teacher CUDA memory smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.agcn import AGCNModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AGCN memory smoke")
    from clip import clip

    device = torch.device("cuda")
    clip_model, _ = clip.load(args.clip_model_path, device="cpu", jit=False)
    model = AGCNModel(
        clip_model.visual.float(), visual_dim=512, embedding_dim=300,
        graph_hidden_dim=1024,
    ).to(device)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    visual_optimizer = torch.optim.Adam(model.visual_encoder.parameters(), lr=1e-4)
    graph_optimizer = torch.optim.Adam(model.graph.parameters(), lr=3e-5)
    images = torch.randn(args.batch_size, 3, 224, 224, device=device)
    embeddings = torch.randn(26, 300, device=device)
    adjacency = torch.eye(26, device=device)
    targets = torch.randint(0, 2, (args.batch_size, 3), device=device).float()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        teacher_logits = teacher(images, adjacency, embeddings)["logits"]
    output = model(images, adjacency, embeddings)
    current = F.binary_cross_entropy_with_logits(output["logits"][:, -3:], targets)
    distill = F.binary_cross_entropy_with_logits(
        output["logits"][:, :-3], torch.sigmoid(teacher_logits[:, :-3])
    )
    relation = F.mse_loss(
        output["graph_nodes"][:-3], teacher.graph_nodes(adjacency, embeddings)[:-3]
    )
    loss = 0.07 * current + 0.93 * distill + 1e5 * relation
    loss.backward()
    visual_optimizer.step()
    graph_optimizer.step()
    print(json.dumps({
        "batch_size": args.batch_size,
        "seen_classes": 26,
        "teacher_present": True,
        "optimizer": "two Adam optimizers",
        "loss_finite": bool(torch.isfinite(loss)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }, indent=2))


if __name__ == "__main__":
    main()
