"""1-minute rolling retrain of AlphaMaster's production factor stack.

This package is a clean rewrite. It does not patch ``model_core``. The factor
vocabulary, StackVM normalization, training label, FTMO score and tanh position
map follow the production path documented in ``REPORT.md``.
"""

__version__ = "1.0.0"
