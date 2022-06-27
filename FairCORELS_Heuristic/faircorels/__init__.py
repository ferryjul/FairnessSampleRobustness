from __future__ import print_function, division, with_statement
from .corels import CorelsClassifier, CorelsBagging
from .utils import load_from_csv, RuleList
from .metrics import ConfusionMatrix, Metric

__version__ = "0.8"

__all__ = ["FairCorelsClassifier", "load_from_csv", "RuleList", "FairCorelsBagging", "ConfusionMatrix", "Metric"]