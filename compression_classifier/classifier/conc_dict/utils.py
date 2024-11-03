import xxhash

from compression_classifier.data.base import BaseDataLoader

def byte_hash(b : bytes) -> int:
    if len(b) < 9:
        return xxhash.xxh32_intdigest(b)
    return xxhash.xxh3_64_intdigest(b)

def choose_k_value(data : BaseDataLoader) -> int:
    unique_tokens = len(set(list(data.X_train[1] + data.X_train[3] + data.X_train[5])))
    if unique_tokens < 20:
        return 10
    elif unique_tokens < 128:
        return 8
    elif unique_tokens < 200:
        return 6
    return 4

def choose_segment_size(dict_size : int, data : BaseDataLoader) -> int:
    for i in range(4, 32):
        if dict_size // 2**i < data.min_class_samples():
            return 2**i
    return None
