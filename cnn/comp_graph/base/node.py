from functools import wraps
import numpy as np

class Node:
    def __init__(self, operation, value, parent):
        self.operation = operation
        self.parent = parent
        self.value = value

    #@staticmethod



def add_operation(op):

    @wraps(op)
    def operation_wrapper(*args):

        def get_val(p): return p.value if isinstance(p, Node) else p

        cargs = [get_val(i) for i in args]
        
        parents = [i for i in args if isinstance(i, Node)]
        val = op(*cargs)

        return Node(op, val, parents)
    return operation_wrapper

def func(*args): return np.sum(args)

a = add_operation(func)
b = add_operation(np.multiply)

print(b(a(1,2),3).value)
