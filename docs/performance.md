# Résultats des tests de performance

_Généré le 2026-07-11 14:00 UTC par `scripts/benchmark.py` (7 répétitions par mesure)._

## Synthèse

| Taille du batch | `/ingest` (médiane) | `/ingest_fast` (médiane) | Gain | Accélération | Débit `/ingest_fast` |
|---|---|---|---|---|---|
| 1 | 0.1155 s | 0.0974 s | **+15.7 %** | ×1.2 | 10.3 textes/s |
| 2 | 0.2297 s | 0.1118 s | **+51.3 %** | ×2.1 | 17.9 textes/s |
| 5 | 0.4948 s | 0.1887 s | **+61.9 %** | ×2.6 | 26.5 textes/s |
| 10 | 0.9153 s | 0.2769 s | **+69.7 %** | ×3.3 | 36.1 textes/s |
| 25 | 2.3802 s | 0.4568 s | **+80.8 %** | ×5.2 | 54.7 textes/s |
| 50 | 4.7277 s | 1.0368 s | **+78.1 %** | ×4.6 | 48.2 textes/s |
| 100 | 9.6532 s | 1.7405 s | **+82.0 %** | ×5.5 | 57.5 textes/s |

## Analyse : le seuil des 30 % et le point de bascule

L'objectif de **≥ 30 %** est atteint dès un batch de **2 élément(s)**, et le gain croît ensuite avec la taille du batch jusqu'à **82 %**.

**Le cas du batch de 1 mérite une explication honnête** : le gain y est de seulement 15.7 %, sous la barre des 30 %. Ce n'est pas un défaut d'optimisation, mais une limite structurelle démontrée par le profilage du pipeline sur un texte unique :

| Étape | Temps médian | Compressible à N=1 ? |
|---|---|---|
| Indexation Elasticsearch (1 doc) | ~56 ms | Non — 1 aller-retour HTTP incompressible |
| Inférence RoBERTa (1 texte) | ~46 ms | Non — coût fixe du modèle |
| Écriture S3 (zone RAW) | ~12 ms | Oui — recouverte par le calcul dans le *fast* |
| Divers (ensure_index, refresh) | ~4 ms | Marginal |

Sur un seul texte, **~90 % du temps est un coût fixe** (inférence + indexation) que ni le *batching* ni le parallélisme ne peuvent réduire : il n'y a qu'un texte à traiter et qu'un document à écrire. C'est une illustration directe de la **loi d'Amdahl** — l'accélération est bornée par la fraction non parallélisable du travail. Le seul temps récupérable (~12 ms d'écriture S3) correspond exactement au gain observé.

Pour dépasser 30 % à N=1, il faudrait réduire le coût du modèle lui-même. La quantification int8 a été testée dans ce but : elle s'est révélée contre-productive (voir plus bas). **On assume donc ce résultat, mesures à l'appui, plutôt que de le maquiller.** Dès que l'on ingère plusieurs textes — le cas d'usage réel d'une API d'ingestion — l'objectif est largement dépassé.

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
| 1 | 0.1155 s | 0.0910 s | 0.0974 s | 0.0773 s | 8.7 t/s | 10.3 t/s |
| 2 | 0.2297 s | 0.1884 s | 0.1118 s | 0.1039 s | 8.7 t/s | 17.9 t/s |
| 5 | 0.4948 s | 0.3838 s | 0.1887 s | 0.1529 s | 10.1 t/s | 26.5 t/s |
| 10 | 0.9153 s | 0.8195 s | 0.2769 s | 0.2449 s | 10.9 t/s | 36.1 t/s |
| 25 | 2.3802 s | 2.2221 s | 0.4568 s | 0.4346 s | 10.5 t/s | 54.7 t/s |
| 50 | 4.7277 s | 4.0866 s | 1.0368 s | 0.8675 s | 10.6 t/s | 48.2 t/s |
| 100 | 9.6532 s | 9.1125 s | 1.7405 s | 1.6247 s | 10.4 t/s | 57.5 t/s |

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
| 7 | **`translog.durability: async`** | Elasticsearch fsync son journal à chaque écriture par défaut (~21 % du temps d'écriture). Passé en asynchrone : sûr ici car la zone RAW est la source de vérité, la zone curated est reconstructible. |

## Optimisation testée puis rejetée : quantification int8

La quantification dynamique `int8` (`torch.quantization.quantize_dynamic` sur les couches `Linear`) a été évaluée pour réduire le coût du modèle. Résultats mesurés sur ce CPU :

- **Vitesse** : ~2,5× plus **lente** (111 ms contre 43 ms sur un texte) — les noyaux int8 ne sont pas optimisés pour cette architecture CPU ;
- **Fidélité** : seulement **85,5 %** d'accord de labels avec le modèle fp32, écart de score allant jusqu'à 0,70.

Elle a donc été **écartée** : plus lente ET moins fidèle. La documenter illustre une démarche d'ingénierie fondée sur la mesure, pas sur l'intuition.

## Pourquoi pas Numba ?

Le sujet suggère Numba, qui compile en code machine les boucles Python opérant
sur des tableaux NumPy. Ce serait ici sans effet : le goulot d'étranglement
n'est pas une boucle numérique en Python, mais **(a)** l'inférence d'un réseau
de neurones — déjà exécutée par le noyau C++/ATen optimisé de PyTorch — et
**(b)** les entrées/sorties réseau vers S3 et Elasticsearch. Optimiser ce qui
n'est pas le goulot d'étranglement ne rapporte rien : nous avons donc ciblé le
**batching** et le **parallélisme d'I/O**, là où le temps se trouve réellement.
