import inspect

class Object:
    def __init__(self, a, b):
        pass

def func(a,b,c = None):
    sig = [str(x).split('=') for x in inspect.stack()[-1]]
    #index = sig.index('code_context')
    print(sig[-2])


func(3,9, Object(1, 2))