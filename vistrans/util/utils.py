from itertools import repeat
import torch
from torch._six import container_abcs


# From PyTorch internals (with little change)
def ntuple(x, n):
    def parse(x, n):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse(x, n)


def save_checkpoint(model, is_best=True, filename='./checkpoint.pth'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        torch.save(model.state_dict(), filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


def load_checkpoint(model, filename='./checkpoint.pth'):
    sd = torch.load(filename, map_location=lambda storage, loc: storage)
    names = set(model.state_dict().keys())
    for n in list(sd.keys()):
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd:
                sd[n+'_raw'] = sd[n]
            del sd[n]
    model.load_state_dict(sd)
