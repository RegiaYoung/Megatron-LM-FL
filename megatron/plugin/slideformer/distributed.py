# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded gradient synchronization for CPU-owned SlideFormer parameters.

Process groups are supplied by Megatron; never infer replica ownership from
world size. Dense parameters use DP×CP groups; experts use expert DP groups.
"""
import torch
import torch.distributed as dist


def synchronize_gradient(gradient, *, group, device, chunk_size_mb, average):
    """Reduce FP32 chunks on the communication device and copy back to CPU."""
    size = dist.get_world_size(group) if group is not None else 1
    if size == 1:
        return
    flat = gradient.view(-1)
    chunk_numel = max(1, int(chunk_size_mb * 1024**2) // 4)
    for start in range(0, flat.numel(), chunk_numel):
        destination = flat[start : start + chunk_numel]
        work = destination.to(device=device, dtype=torch.float32)
        dist.all_reduce(work, group=group)
        if average:
            work.div_(size)
        destination.copy_(work.to(device=destination.device))
