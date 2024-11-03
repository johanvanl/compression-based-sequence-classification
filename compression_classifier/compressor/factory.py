from enum import Enum, auto

from compression_classifier.compressor.base import BaseCompressor

from compression_classifier.compressor.lz4 import Lz4Compressor
from compression_classifier.compressor.zlib import ZlibCompressor
from compression_classifier.compressor.zstd import ZstdCompressor

class ECompressor(Enum):

    LZ4 = auto()
    ZLIB = auto()
    ZSTD = auto()

def max_lvl_compressor_factory(e_compressor : ECompressor) -> BaseCompressor:
    match e_compressor:
        case ECompressor.LZ4:
            return Lz4Compressor(level=12)
        case ECompressor.ZLIB:
            return ZlibCompressor(level=9)
        case ECompressor.ZSTD:
            return ZstdCompressor(level=22)
    raise ValueError('Unknown Compressor!')

def max_lvl_dict_compressor_factory(e_compressor : ECompressor, dict : bytes) -> BaseCompressor:
    match e_compressor:
        case ECompressor.LZ4:
            return Lz4Compressor(level=12, dict=dict)
        case ECompressor.ZLIB:
            return ZlibCompressor(level=9, dict=dict)
        case ECompressor.ZSTD:
            return ZstdCompressor(level=22, dict=dict)
    raise ValueError('Unknown Compressor!')

def compressor_factory(e_compressor : ECompressor, level : int, dict : bytes) -> BaseCompressor:
    match e_compressor:
        case ECompressor.LZ4:
            return Lz4Compressor(level=level, dict=dict)
        case ECompressor.ZLIB:
            return ZlibCompressor(level=level, dict=dict)
        case ECompressor.ZSTD:
            return ZstdCompressor(level=level, dict=dict)
    raise ValueError('Unknown Compressor!')
