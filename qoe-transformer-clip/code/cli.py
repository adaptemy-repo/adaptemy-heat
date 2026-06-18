import json

from exp import ex
from args import get_args

from utils import stringify_dict
from train import train as _train
from train_adapt import train as _train_adapt
from train_adapt_single import train as _train_adapt_single
from train_salivqa import train as _train_salivqa
from demo import demo as _demo
from demo_adapt import demo_adapt as _demo_adapt
from demo_salivqa import demo_salivqa as _demo_salivqa
from demo_brisque import demo_brisque as _demo_brisque

@ex.command
def train(_config):
    result = _train()

    return 0

@ex.command
def train_adapt(_config):
    result = _train_adapt()

    return 0

@ex.command
def train_adapt_single(_config):
    result = _train_adapt_single()

    return 0

@ex.command
def train_salivqa(_config):
    result = _train_salivqa()

    return 0

@ex.command
def demo(_config):
    result = _demo()

    return 0

@ex.command
def demo_adapt(_config):
    result = _demo_adapt()

    return 0

@ex.command
def demo_salivqa(_config):
    result = _demo_salivqa()

    return 0

@ex.command
def demo_brisque(_config):
    result = _demo_brisque()

    return 0

@ex.option_hook
def update_args(options):
    args = get_args(options)

    print(json.dumps(stringify_dict(args), indent=4))
    ex.add_config(args)
    return options


@ex.automain
def run():
    train()
