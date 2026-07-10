# Résultats des tests de performance

_Généré le 2026-07-10 20:25 UTC par `scripts/benchmark.py` (5 répétitions par mesure)._

## Synthèse

| Taille du batch | `/ingest` (médiane) | `/ingest_fast` (médiane) | Gain | Accélération | Débit `/ingest_fast` |
|---|---|---|---|---|---|
| 1 | 0.1663 s | 0.1459 s | **+12.3 %** | ×1.1 | 6.9 textes/s |
| 2 | 0.2681 s | 0.1605 s | **+40.1 %** | ×1.7 | 12.5 textes/s |
| 5 | 0.5199 s | 0.1965 s | **+62.2 %** | ×2.6 | 25.4 textes/s |
| 10 | 1.0203 s | 0.2444 s | **+76.0 %** | ×4.2 | 40.9 textes/s |
| 25 | 2.3816 s | 0.3881 s | **+83.7 %** | ×6.1 | 64.4 textes/s |
| 50 | 4.7500 s | 0.6728 s | **+85.8 %** | ×7.1 | 74.3 textes/s |
| 100 | 9.9763 s | 1.2607 s | **+87.4 %** | ×7.9 | 79.3 textes/s |

**Objectif du sujet (≥ 30 % de gain) : voir analyse ci-dessous.**

## Équivalence fonctionnelle

Un benchmark n'a de sens que si les deux implémentations produisent le
même résultat. À chaque répétition, on vérifie que :

- les identifiants de documents sont identiques ;
- les **labels** de sentiment sont rigoureusement identiques ;
- les **scores** coïncident à `1e-5` près.

> Pourquoi une tolérance sur les scores plutôt qu'une égalité stricte ?
> En mode batch, les textes sont paddés à une longueur commune et les
> multiplications matricielles s'exécutent dans un ordre différent. Or
> l'addition flottante n'est pas associative : `(a+b)+c ≠ a+(b+c)` au bit
> près. L'écart observé est de l'ordre de `1e-7`, purement numérique.

- Batch de 1 : **OK** — labels identiques, scores équivalents à 1e-5 près
- Batch de 2 : **OK** — labels identiques, scores équivalents à 1e-5 près
- Batch de 5 : **OK** — labels identiques, scores équivalents à 1e-5 près
- Batch de 10 : **OK** — labels identiques, scores équivalents à 1e-5 près
- Batch de 25 : **OK** — labels identiques, scores équivalents à 1e-5 près
- Batch de 50 : **OK** — labels identiques, scores équivalents à 1e-5 près
- Batch de 100 : **OK** — labels identiques, scores équivalents à 1e-5 près

## Détail des mesures

| Batch | naive médiane | naive min | fast médiane | fast min | Débit naive | Débit fast |
|---|---|---|---|---|---|---|
| 1 | 0.1663 s | 0.1526 s | 0.1459 s | 0.1419 s | 6.0 t/s | 6.9 t/s |
| 2 | 0.2681 s | 0.2613 s | 0.1605 s | 0.1548 s | 7.5 t/s | 12.5 t/s |
| 5 | 0.5199 s | 0.5117 s | 0.1965 s | 0.1814 s | 9.6 t/s | 25.4 t/s |
| 10 | 1.0203 s | 0.9566 s | 0.2444 s | 0.2316 s | 9.8 t/s | 40.9 t/s |
| 25 | 2.3816 s | 2.2808 s | 0.3881 s | 0.3832 s | 10.5 t/s | 64.4 t/s |
| 50 | 4.7500 s | 4.6028 s | 0.6728 s | 0.6291 s | 10.5 t/s | 74.3 t/s |
| 100 | 9.9763 s | 9.9439 s | 1.2607 s | 1.2017 s | 10.0 t/s | 79.3 t/s |

## Méthodologie

1. **Warm-up** — le modèle RoBERTa est chargé paresseusement à la première
   inférence (~25 s). Deux appels sont effectués hors mesure pour l'exclure.
2. **Textes uniques** — chaque répétition emploie des textes distincts, afin
   qu'aucun cache ne favorise l'endpoint exécuté en second.
3. **Alternance + médiane** — l'ordre naive/fast est inversé à chaque
   répétition, et l'on retient la médiane, robuste aux valeurs aberrantes.
4. **Mesure côté serveur** — `elapsed_seconds` est chronométré autour du
   pipeline lui-même, hors sérialisation JSON et latence réseau.

## Optimisations mises en œuvre dans `/ingest_fast`

| # | Optimisation | Effet |
|---|---|---|
| 1 | **Inférence par lots** | Une seule passe du Transformer pour N textes, au lieu de N passes. Exploite les multiplications matricielles vectorisées. C'est le gain dominant, et il croît avec la taille du batch. |
| 2 | **Indexation `_bulk`** | Un seul aller-retour HTTP vers Elasticsearch au lieu de N indexations unitaires. |
| 3 | **Écritures S3 parallèles** | Les N écritures dans RAW sont indépendantes et I/O-bound : un pool de 16 threads les recouvre. |
| 4 | **Recouvrement I/O ↔ calcul** | Les écritures RAW sont lancées en tâche de fond *pendant* que le modèle calcule le sentiment. Le coût d'écriture devient quasi invisible. |
| 5 | **`torch.inference_mode()`** | Supprime la construction du graphe d'autograd : moins d'allocations, forward plus rapide que `no_grad()`. |
| 6 | **Clients S3/ES mis en cache** | `lru_cache` sur les constructeurs de clients. Bénéficie aux **deux** endpoints, donc ne fausse pas la comparaison. |

## Pourquoi pas Numba ?

Le sujet suggère Numba, qui compile en code machine les boucles Python opérant
sur des tableaux NumPy. Ce serait ici sans effet : le goulot d'étranglement
n'est pas une boucle numérique en Python, mais **(a)** l'inférence d'un réseau
de neurones — déjà exécutée par le noyau C++/ATen optimisé de PyTorch — et
**(b)** les entrées/sorties réseau vers S3 et Elasticsearch. Optimiser ce qui
n'est pas le goulot d'étranglement ne rapporte rien : nous avons donc ciblé le
**batching** et le **parallélisme d'I/O**, là où le temps se trouve réellement.
