import zstandard as zstd
import pyzstd

from compression_classifier.compressor.base import BaseCompressor

class ZstdCompressor(BaseCompressor):
    
    def __init__(self, level : int = 3, dict : bytes = None) -> None:
        '''
        Args:
            level (int, optional): The compression level between 1 and 22 (incl).
                                   Defaults to 3.
        '''
        super().__init__(level, dict)
        assert level >= 1 and level <= 22, 'Level is out of range!'
        self.final = False

    def _finalize(self) -> None:
        self.final = True
        if self.dict is not None:
            # Use magic header to check whether full or raw dict
            zstd_dict = tuple(self.dict[:4]) == (55, 164, 48, 236)
            if zstd_dict:
                zstd_dict = zstd.ZstdCompressionDict(self.dict, dict_type=zstd.DICT_TYPE_FULLDICT)
            else:
                zstd_dict = zstd.ZstdCompressionDict(self.dict, dict_type=zstd.DICT_TYPE_RAWCONTENT)
            zstd_dict.precompute_compress(level=self.level)
            self.compressor = zstd.ZstdCompressor(dict_data=zstd_dict, level=self.level,
                                                  write_checksum=False, write_content_size=False,
                                                  write_dict_id=False)
        else:
            self.compressor = zstd.ZstdCompressor(level=self.level, write_checksum=False,
                                                  write_content_size=False, write_dict_id=False)

    def _compress(self, b : bytes) -> bytes:
        if not self.final:
            self._finalize()
        return self.compressor.compress(b)
    
    def compressed_len(self, b : bytes) -> int:
        # Remove the Magic Number
        # https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md
        return len(self._compress(b)) - 4

def train_zstd_dict(samples : list[bytes], dict_size : int, level : int) -> bytes:
    '''
    Using samples train a zstd dictionary.

    Args:
        samples (list[bytes]): Samples that represent what will be compressed.
        dict_size (int): The byte size of the dictionary.
        level (int): What level compression will be used when compressing
                     with the dictionary.
    Returns:
        bytes: A full zstd dictionary.
    '''
    # Add space for the entropy tables, so that the dictionary is
    # comparable to the finalized dictionary.
    di = zstd.train_dictionary(dict_size=dict_size + 256, samples=samples, level=level)
    return di.as_bytes()

def finalize_dict_for_zstd(dict : bytes, samples : list[bytes], level : int) -> bytes:
    '''
    Provided with a precomputed dictionary, use the samples
    to add the entropy tables for a full zstd dictionary.

    Args:
        dict (bytes): The precomputed dictionary.
        samples (list[bytes]): Samples that represent what will be compressed.
        level (int): What level compression will be used when compressing
                     with the dictionary.
    Returns:
        bytes: A full zstd dictionary.
    '''
    # Add space for the entropy tables
    return pyzstd.finalize_dict(pyzstd.ZstdDict(dict, is_raw=True),
                                samples=samples, level=level,
                                dict_size=len(dict) + 512).dict_content

if __name__ == '__main__':
    c = ZstdCompressor()
    print(c._compress(b'a'))
