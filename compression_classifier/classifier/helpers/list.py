import numpy as np

def split_into_classes(X : list[bytes], y : np.ndarray) -> list[list[bytes]]:
    num_classes = len(np.unique(y))
    cls_samples = [[] for _ in range(num_classes)]
    for i in range(len(y)):
        cls_samples[y[i]].append(X[i])
    return cls_samples

def evenly_batch(X : list[bytes], segments : int) -> list[list[bytes]]:
    size = len(X) // segments
    remainder = len(X) % segments

    segment_samples = []

    start = 0
    for segment in range(segments):
        end = start + size + (1 if segment < remainder else 0)
        segment_samples.append(X[start:end])
        start = end

    return segment_samples
