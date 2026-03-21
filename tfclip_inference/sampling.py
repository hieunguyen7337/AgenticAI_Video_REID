from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def dense_sample_frames(frames: Sequence[T], seq_len: int) -> list[list[T]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if not frames:
        raise ValueError("frames must not be empty")

    num_frames = len(frames)
    if num_frames <= seq_len:
        clip = list(frames)
        while len(clip) < seq_len:
            clip.append(clip[-1])
        return [clip]

    clips: list[list[T]] = []
    cur_index = 0
    while num_frames - cur_index > seq_len:
        clips.append(list(frames[cur_index:cur_index + seq_len]))
        cur_index += seq_len

    tail = list(frames[cur_index:])
    while len(tail) < seq_len:
        tail.append(tail[-1])
    clips.append(tail)
    return clips
