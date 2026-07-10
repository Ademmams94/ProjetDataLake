"""
Contournement d'un conflit de DLL sous Windows.

PyTorch et PyArrow (utilisé par pandas pour le Parquet) embarquent chacun
leur propre runtime OpenMP (`libiomp5md.dll`). Si PyArrow est chargé en
premier, l'initialisation de `c10.dll` de PyTorch échoue :

    OSError: [WinError 1114] Une routine d'initialisation d'une
    bibliothèque de liens dynamiques (DLL) a échoué.

La parade : charger PyTorch AVANT pandas/pyarrow.

Usage — l'importer en TOUT PREMIER dans les modules qui utilisent les deux :

    from src.common import torch_first  # noqa: F401  (doit précéder pandas)
    import pandas as pd

L'import est tolérant : si PyTorch n'est pas installé (mode VADER seul),
le module ne fait rien et n'empêche pas le reste de tourner.
"""

try:
    import torch  # noqa: F401  — import à effet de bord, volontairement inutilisé
except ImportError:  # pragma: no cover - environnement sans Deep Learning
    torch = None
