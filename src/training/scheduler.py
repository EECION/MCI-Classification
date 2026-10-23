"""5-epoch linear warm-up followed by cosine annealing to `min_lr` (paper Section 2.4)."""
import math

from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine(optimizer, warmup_epochs, total_epochs, min_lr):
    def make_lambda(min_ratio):
        def fn(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cos
        return fn

    lambdas = [make_lambda(min_lr / g["lr"] if g["lr"] > 0 else 0.0)
               for g in optimizer.param_groups]
    return LambdaLR(optimizer, lambdas)
