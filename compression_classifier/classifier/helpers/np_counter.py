from collections import Counter

from compression_classifier.classifier.helpers.system_id_bidict import StaticNoCopySystemIdBiDict

import numpy as np

class NPCounter:

    def __init__(self, bidict : StaticNoCopySystemIdBiDict,
                 dtype : np.dtype = np.float64) -> None:
        self.bidict = bidict
        self.array = np.zeros(len(bidict), dtype=dtype)

    def add_counter(self, cnt : Counter) -> None:
        for key in cnt:
            if key in self.bidict:
                self.array[self.bidict.encode(key)] += cnt[key]

    def high_values(self, size : int) -> list[tuple]:
        li = []
        for i in np.flip(np.argsort(self.array))[:size]:
            li.append((self.bidict.decode(i), self.array[i]))
        return li
    
    def low_values(self, size : int) -> list[tuple]:
        li = []
        for i in np.argsort(self.array)[:size]:
            li.append((self.bidict.decode(i), self.array[i]))
        return li
    
    def build_positive_counter(self) -> Counter:
        cnt = Counter()
        for i in np.where(self.array > 0)[0]:
            cnt[self.bidict.decode(i)] = self.array[i]
        return cnt
    
    def __len__(self) -> int:
        return len(self.array)
    
    def __contains__(self, key) -> bool:
        return key in self.bidict

    def __getitem__(self, key):
        if key not in self.bidict:
            return 0
        return self.array[self.bidict.encode(key)]
    
    def __setitem__(self, key, value):
        self.array[self.bidict.encode(key)] = value

if __name__ == '__main__':
    bidi = StaticNoCopySystemIdBiDict(set(['a', 'b', 'c']))
    cnt = NPCounter(bidi)
    print(cnt.array)
    cnt['a'] = 5
    cnt['b'] += 5
    print(cnt.array)
