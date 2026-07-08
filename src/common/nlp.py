"""
Fonctions NLP partagées : nettoyage de texte et analyse de sentiment.

Note importante sur le nettoyage : on NE met PAS le texte en minuscules et on
GARDE la ponctuation. Pourquoi ? Parce que VADER (notre moteur de sentiment)
exploite justement les MAJUSCULES (emphase) et la ponctuation (« super!!! »)
pour ajuster son score. Un nettoyage trop agressif dégraderait la qualité.
"""
from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """
    Nettoyage léger et sûr :
    - décode les entités HTML (&amp; -> &) et retire les balises (<p>, <a>…)
    - supprime les URLs (bruit pour l'analyse de sentiment)
    - normalise les espaces multiples
    On conserve casse et ponctuation (utiles pour VADER).
    """
    if not text:
        return ""
    text = html.unescape(text)          # &#x27; -> '
    text = HTML_TAG_RE.sub(" ", text)   # <br> -> espace
    text = URL_RE.sub(" ", text)        # retire les liens
    text = WHITESPACE_RE.sub(" ", text) # espaces multiples -> un seul
    return text.strip()
