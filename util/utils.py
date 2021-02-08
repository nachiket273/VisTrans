from itertools import repeat
from torch._six import container_abcs


# From PyTorch internals (with little change)
def ntuple(x, n):
    def parse(x, n):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse(x, n)
