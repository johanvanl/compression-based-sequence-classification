from abc import ABC, abstractmethod
from collections import Counter

class BaseNoCopySystemIdBiDict(ABC):
    '''
    SystemIdBiDict is a Data Structure implementing a bidirectional dictionary
    where the value cannot be specified but is instead assigned a System Id which is
    zero indexed.

    Enabling you to loop over the System Ids from zero to the size.

    Allowing estimators to share data when invoked via
    GridSearchCV, which copies all objects.
    '''

    def __init__(self) -> None:
        super().__init__()
        self.last_system_id = -1
        self.di = {}
        self.inverse_di = {}

    def key_set(self) -> set:
        return self.di.keys()

    @abstractmethod
    def encode(self, key) -> int:
        '''
        Given a key, return the corresponding System Id.
        '''
        pass
    
    def decode(self, system_id : int):
        '''
        Given a System Id, return the corresponding key.
        '''
        if system_id not in self.inverse_di:
            raise KeyError(f"The System Id ('{system_id}') is not assigned!")
        return self.inverse_di[system_id]
    
    def __len__(self) -> int:
        return len(self.di)
    
    def __contains__(self, key) -> bool:
        return key in self.di
    
    def __copy__(self):
        return self
    
    def __deepcopy__(self, memo):
        return self

class StaticNoCopySystemIdBiDict(BaseNoCopySystemIdBiDict):

    def __init__(self, keys : set | Counter, min_cnt : int = 0):
        super().__init__()

        assert isinstance(keys, set) or isinstance(keys, Counter), \
            f'Keys needs to be either a set or Counter, not {type(keys)}!'        

        if isinstance(keys, set):
            for key in keys:
                self.last_system_id += 1
                self.di[key] = self.last_system_id
                self.inverse_di[self.last_system_id] = key
            return

        # Counter with threshold check
        for key in keys:
            if keys[key] >= min_cnt:
                self.last_system_id += 1
                self.di[key] = self.last_system_id
                self.inverse_di[self.last_system_id] = key

    def encode(self, key) -> int:
        if key not in self.di:
            raise KeyError(f"The Key ('{key}') was not assigned!")
        return self.di[key]

class DynamicNoCopySystemIdBiDict(BaseNoCopySystemIdBiDict):

    def __init__(self) -> None:
        super().__init__()

    def encode(self, key) -> int:
        if key not in self.di:
            self.last_system_id += 1
            self.di[key] = self.last_system_id
            self.inverse_di[self.last_system_id] = key
            return self.last_system_id
        return self.di[key]
