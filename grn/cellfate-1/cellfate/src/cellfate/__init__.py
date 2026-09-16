"""cellfate: GPU-native weighted-ensemble stochastic simulation of gene regulatory networks.

Public API::

    from cellfate import models
    from cellfate.model import ReactionNetwork, Reaction, Modifier, ACTIVATION, REPRESSION
    from cellfate.engine.torch_backend import TorchEngine
    from cellfate.we import BinScheme, WEConfig, WeightedEnsemble
"""

__version__ = "0.1.0"
