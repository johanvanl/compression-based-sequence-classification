import struct
import lz4.block as lz4
from compression_classifier.compressor.base import BaseCompressor

# Padding forces allows the end of block sequence to include
# some of the provided bytes, allowing a better estimate
# (https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md)
PADDING_LEN = 5
PADDING = PADDING_LEN * int(0).to_bytes(length=1, byteorder='big')

class Lz4Compressor(BaseCompressor):
    
    def __init__(self, level : int = 9, dict : bytes = None) -> None:
        '''
        Args:
            level (int, optional): The compression level between 1 and 12 (incl).
                                   Defaults to 9.
        '''
        super().__init__(level, dict)
        assert level >= 1 and level <= 12, 'Level is out of range!'

    def _compress(self, b : bytes) -> bytes:
        b += PADDING
        if self.dict is not None:
            comp_b = lz4.compress(b, dict=self.dict,
                                  mode='high_compression',
                                  compression=self.level,
                                  store_size=False)
        else:
            comp_b = lz4.compress(b, compression=self.level,
                                  mode='high_compression',
                                  store_size=False)
        # Remove the PADDING
        return comp_b[:-PADDING_LEN]
    
    def compressed_len(self, b : bytes) -> int:
        return len(self._compress(b))

    def get_lz4_sequences(self, b : bytes) -> list[tuple]:
        '''
        Obtain the sequences that make up a LZ4 compression.

        Args:
            b (bytes): The LZ4 compressed bytes.

        Returns:
            list[tuple]: The list of sequences that make up the compression
                         with a sequence tuple defined as
                         (literals, offset, math length)
        '''
        sequences = []
        idx = 0
        while idx < len(b):
            # New Sequence
            while True:
                # Read Token
                lit_len = ( b[idx] >> 4 ) & 0x0F
                match_len = ( b[idx] & 0x0F ) + 4
                idx += 1
                
                # Read further for Literals Length
                if lit_len == 15:
                    while b[idx] == 255:
                        lit_len += 255
                        idx += 1
                    lit_len += b[idx]
                    idx += 1

                # Read Literals
                lit = b[idx:idx+lit_len]
                idx += lit_len

                # The last sequence, from where the PADDING
                # has been removed
                if idx >= len(b):
                    if len(lit) > 0:
                        sequences.append((lit, 0, 0))
                    break
                
                # Read Offset
                offset = struct.unpack('<H', b[idx:idx+2])[0]
                idx += 2

                # Read Further for Match Length
                if match_len == 19:
                    while b[idx] == 255:
                        match_len += 255
                        idx += 1
                    match_len += b[idx]
                    idx += 1

                sequences.append((lit, offset, match_len))

        return sequences

    def get_dict_matches(self, sequences : list[tuple]) -> list[tuple]:
        '''
        Get only the matches in the dictionary.

        Args:
            sequences (list[tuple]): The sequences obtained from 'get_lz4_sequences'.

        Returns:
            list[tuple]: The dictionary matches as a list of tuples, where the tuple is
                         (zero indexed position in dictionary, match length)
        '''
        di_matches = []
        cursor = 0
        for literal, offset, match_len in sequences:
            cursor += len(literal)
            if cursor < offset:
                di_matches.append((self.dict_size - abs(cursor - offset), match_len))
            cursor += match_len
        return di_matches
