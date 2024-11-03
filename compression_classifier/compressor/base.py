from abc import ABC, abstractmethod

class BaseCompressor(ABC):
    '''
    Abstract Base Compressor.
    '''

    def __init__(self, level : int, dict : bytes = None) -> None:
        '''
        Args:
            level (int, optional): The compression level.
            dict (bytes, optional): A dictionary. Defaults to None.
        '''
        super().__init__()
        self.level = level
        self.dict = dict
        
        self.dict_size = None
        if self.dict is not None:
            self.dict_size = len(self.dict)

    def set_dict(self, dict : bytes) -> None:
        self.dict = dict
    
    @abstractmethod
    def _compress(self, b : bytes) -> bytes:
        '''
        Given a bytes object, return the compressed bits.

        Args:
            b (bytes): The bytes to compress.

        Returns:
            bytes: The compressed bytes
        '''
        pass

    def compress(self, b : bytes) -> bytes:
        '''
        Obtain the compressed bits from the compressor.

        Args:
            b (bytes): The bytes to compress

        Returns:
            bytes: The compressed bytes.
        '''
        return self._compress(b)
    
    def compress_str(self, s : str) -> bytes:
        '''
        Obtain the compressed bits from the compressor.

        Args:
            s (str): The string to compress

        Returns:
            bytes: The compressed bytes.
        '''
        return self._compress(s.encode(encoding='utf-8'))
    
    @abstractmethod
    def compressed_len(self, b : bytes) -> int:
        '''
        Obtain the compressed length.

        Args:
            b (bytes): The bytes to compress.

        Returns:
            int: The length of the compression.
        '''
        pass

class CompressorList:
    '''
    Holds a list of BaseCompressor's
    '''

    def __init__(self) -> None:
        self.compressors = []

    def add(self, compressor : BaseCompressor):
        '''
        Add a BaseCompressor object.
        '''
        self.compressors.append(compressor)

    def get(self, idx : int) -> BaseCompressor:
        '''
        Get the compressor at the provided index.

        Args:
            idx (int): The compressor index.

        Returns:
            BaseCompressor: The compressor.
        '''
        return self.compressors[idx]
    
    def __len__(self) -> int:
        return len(self.compressors)
