from . import (ddp)

def build_model(cfg, classnames):

    model = ddp(cfg, classnames)
    return model