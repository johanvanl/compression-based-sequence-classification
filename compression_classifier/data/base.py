from abc import ABC
import base64

import numpy as np

import pyarrow.compute as pc
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split

class BaseDataLoader(ABC):
    '''
    Abstract Base Data Loader Class. Classes that implement the
    BaseDataLoader are expected to provide zero-indexed class labels.
    '''

    def __init__(self) -> None:
        super().__init__()
        self.class_table = None
        self.class_label_di = None
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def min_class_samples(self) -> int:
        return pc.min(self.class_table['train_count']).as_py()
    
    def avg_class_samples(self) -> int:
        return pc.mean(self.class_table['train_count']).as_py()
    
    def median_class_samples(self) -> int:
        return pc.approximate_median(self.class_table['train_count']).as_py()
    
    def max_class_samples(self) -> int:
        return pc.max(self.class_table['train_count']).as_py()
    
    def min_sample_length(self) -> int:
        return pc.min(self.class_table['min_length']).as_py()

    def avg_sample_length(self) -> int:
        return int(pc.mean(self.class_table['avg_length']).as_py())
    
    def median_sample_length(self) -> int:
        return int(pc.approximate_median(self.class_table['avg_length']).as_py())
    
    def max_sample_length(self) -> int:
        return pc.max(self.class_table['max_length']).as_py()
    
    def std_sample_length(self) -> int:
        return int(pc.mean(self.class_table['std_length']).as_py())

    def class_count(self) -> int:
        '''
        Return the class count.
        '''
        return len(self.class_label_di)

    def class_label(self, idx : int) -> str:
        '''
        Return the class label.

        Args:
            idx (int): Class Index.
        '''
        return self.class_label_di[idx]

    def trim(self, train_size : float = -1, test_size : float = -1, random_state : int = 8) -> None:
        '''
        Given the datasets, train/test, we trim them down to specified sizes
        (float for percentage, int for count). Keeping the original class distribution.
        '''
        if train_size > 0:
            self.X_train, _, self.y_train, _ = train_test_split(self.X_train, self.y_train, train_size=train_size,
                                                                random_state=random_state, shuffle=True,
                                                                stratify=self.y_train)
            self.y_train = np.array(self.y_train, dtype=np.int32)

        if test_size > 0:
            self.X_test, _, self.y_test, _ = train_test_split(self.X_test, self.y_test, train_size=test_size,
                                                              random_state=random_state, shuffle=True,
                                                              stratify=self.y_test)
            self.y_test = np.array(self.y_test, dtype=np.int32)

    def __str__(self) -> str:
        s = []
        s.append(f'Total Train: {pc.sum(self.class_table["train_count"]).as_py()}, Total Train: {pc.sum(self.class_table["test_count"]).as_py()}')
        for batch in self.class_table.to_batches(1):
            di = batch.to_pydict()
            s.append(f'"{di["name"][0]}" ({di["idx"][0]}), Train: {di["train_count"][0]}, Test: {di["test_count"][0]}, Min: {di["min_length"][0]}, Avg: {di["avg_length"][0]}±{di["std_length"][0]}, Max: {di["max_length"][0]}')
        
        s.append(f'Avg Length: {int(pc.mean(self.class_table["avg_length"]).as_py())}±{int(pc.mean(self.class_table["std_length"]).as_py())}')
        return '\n'.join(s)

class StringDataLoader(BaseDataLoader):

    def __init__(self, folder : str) -> None:
        super().__init__()

        self.class_table = pq.read_table(f'{folder}classes.parquet')
        self.class_label_di = dict(zip(self.class_table['idx'].to_pylist(),
                                       self.class_table['name'].to_pylist()))

        train_data = pq.read_table(f'{folder}train.parquet')
        for x in train_data['data']:
            self.X_train.append(x.as_py().encode(encoding='utf-8'))
        for y in train_data['label']:
            self.y_train.append(y.as_py())

        test_data = pq.read_table(f'{folder}test.parquet')
        for x in test_data['data']:
            self.X_test.append(x.as_py().encode(encoding='utf-8'))
        for y in test_data['label']:
            self.y_test.append(y.as_py())

class Base64DataLoader(BaseDataLoader):

    def __init__(self, folder : str) -> None:
        super().__init__()

        self.class_table = pq.read_table(f'{folder}classes.parquet')
        self.class_label_di = dict(zip(self.class_table['idx'].to_pylist(),
                                       self.class_table['name'].to_pylist()))

        train_data = pq.read_table(f'{folder}train.parquet')
        for x in train_data['data']:
            self.X_train.append(base64.b64decode(x.as_py().encode(encoding='ascii')))
        for y in train_data['label']:
            self.y_train.append(y.as_py())

        test_data = pq.read_table(f'{folder}test.parquet')
        for x in test_data['data']:
            self.X_test.append(base64.b64decode(x.as_py().encode(encoding='ascii')))
        for y in test_data['label']:
            self.y_test.append(y.as_py())
