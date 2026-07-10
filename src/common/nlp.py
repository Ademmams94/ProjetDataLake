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
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

# Seuils standards recommandés par les auteurs de VADER pour convertir
# le score continu `compound` (dans [-1, 1]) en étiquette discrète.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


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


# ---------------------------------------------------------------
#  Analyse de sentiment (VADER)
# ---------------------------------------------------------------
# VADER = Valence Aware Dictionary and sEntiment Reasoner.
# C'est un modèle à base de lexique + règles, conçu POUR le texte social
# (il gère les emojis, l'argot, les majuscules d'emphase, les négations
# et les intensificateurs type "very"). Avantages ici : rapide, offline,
# déterministe — donc reproductible, ce qui compte pour un data lake.

@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """
    Instancie l'analyseur UNE SEULE FOIS (il charge un lexique de ~7500 mots).
    Le cache évite de le recharger à chaque appel : gain de perf majeur.
    """
    return SentimentIntensityAnalyzer()


def label_from_score(compound: float) -> str:
    """Convertit le score continu en étiquette : positive / neutral / negative."""
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def score_sentiment(text: str) -> tuple[str, float]:
    """Analyse un texte. Renvoie (étiquette, score compound dans [-1, 1])."""
    if not text:
        return "neutral", 0.0
    compound = get_analyzer().polarity_scores(text)["compound"]
    return label_from_score(compound), compound


def score_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Analyse une liste de textes (version séquentielle, référence de perf)."""
    return [score_sentiment(t) for t in texts]


# ---------------------------------------------------------------
#  Analyse de sentiment (Transformer RoBERTa) — Deep Learning
# ---------------------------------------------------------------
# Modèle : cardiffnlp/twitter-roberta-base-sentiment-latest
# Un RoBERTa pré-entraîné sur ~124M de tweets puis fine-tuné sur la tâche
# de sentiment à 3 classes. Contrairement à VADER, il APPREND le contexte
# (sarcasme, négations complexes, expressions idiomatiques).
#
# Les imports lourds (torch, transformers) sont faits à l'intérieur de la
# fonction : le module reste importable et léger si on n'utilise que VADER.

@lru_cache(maxsize=1)
def get_transformer():
    """Charge tokenizer + modèle UNE SEULE FOIS (coûteux : ~500 Mo)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.common.config import settings

    tokenizer = AutoTokenizer.from_pretrained(settings.sentiment_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.sentiment_model)
    model.eval()          # mode inférence : désactive dropout & co
    torch.set_grad_enabled(False)   # pas de gradients : on ne s'entraîne pas
    return tokenizer, model


def score_batch_transformer(texts: list[str],
                            batch_size: int | None = None) -> list[tuple[str, float]]:
    """
    Analyse une liste de textes avec le Transformer, PAR LOTS.

    Le batching est la clé de la performance : passer 32 textes d'un coup
    dans le réseau exploite les multiplications matricielles vectorisées,
    au lieu de 32 passes séquentielles. C'est le levier qu'on exploitera
    pour l'endpoint /ingest_fast.

    Renvoie (label, score) où score = P(positif) - P(négatif), dans [-1, 1],
    afin de garder la même sémantique que le score `compound` de VADER.
    """
    import torch

    from src.common.config import settings

    if not texts:
        return []
    batch_size = batch_size or settings.sentiment_batch_size
    tokenizer, model = get_transformer()

    # Le modèle expose ses propres noms de classes : on les lit plutôt
    # que de les coder en dur (robustesse si on change de modèle).
    id2label = {i: lbl.lower() for i, lbl in model.config.id2label.items()}
    label2id = {lbl: i for i, lbl in id2label.items()}
    pos_i, neg_i = label2id["positive"], label2id["negative"]

    results: list[tuple[str, float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = [t or "" for t in texts[start:start + batch_size]]
        encoded = tokenizer(chunk, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        for row in probs:
            label = id2label[int(row.argmax())]
            score = float(row[pos_i] - row[neg_i])
            results.append((label, score))
    return results


def score_texts(texts: list[str]) -> list[tuple[str, float]]:
    """
    Point d'entrée unique du scoring, piloté par la config.
    Permet de basculer entre les deux moteurs sans toucher aux pipelines.
    """
    from src.common.config import settings

    if settings.sentiment_engine == "vader":
        return score_batch(texts)
    return score_batch_transformer(texts)
