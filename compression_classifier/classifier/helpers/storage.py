
class NoCopyDict:
    '''
    Wrapper for a dictionary that returns itself
    upon copy and deepcopy.
    Allowing estimators to share data when invoked via
    GridSearchCV, which copies all objects.
    '''

    def __init__(self) -> None:
        self.di = {}

    def __contains__(self, key) -> bool:
        return key in self.di
    
    def __getitem__(self, key):
        return self.di[key] 
    
    def __setitem__(self, key, value):
        self.di[key] = value

    def __copy__(self):
        return self
    
    def __deepcopy__(self, memo):
        return self
