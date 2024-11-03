import zlib

from compression_classifier.compressor.base import BaseCompressor

class ZlibCompressor(BaseCompressor):
    
    def __init__(self, level : int = 6, dict : bytes = None) -> None:
        '''
        Args:
            level (int, optional): The compression level between 1 and 9 (incl).
                                   Defaults to 6.
        '''
        super().__init__(level, dict)
        assert level >= 1 and level <= 9, 'Level is out of range!'     

    def _compress(self, b : bytes) -> bytes:
        if self.dict is not None:
            c = zlib.compressobj(zdict=self.dict, level=self.level,
                                 wbits=-15, memLevel=9,
                                 strategy=zlib.Z_DEFAULT_STRATEGY)
        else:
            c = zlib.compressobj(level=self.level, wbits=-15,
                                 memLevel=9, strategy=zlib.Z_DEFAULT_STRATEGY)

        comp_b = c.compress(b)
        comp_b += c.flush(zlib.Z_FINISH)

        return comp_b
    
    def compressed_len(self, b : bytes) -> int:
        return len(self._compress(b))
