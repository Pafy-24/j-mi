# Reevaluare MI și LSR pe corpusurile 2k+ — 23.08.2026

## Verdict executiv

Implementarea MI funcționează tehnic pe RTX 5070 Laptop 8 GB: toate cele șase
fațete au fost antrenate, toate identity-check-urile au avut diferență maximă de
logits `0.0`, dependența `coder.python -> coder` a fost rezolvată corect, iar
cele trei stări de inferență au încăput în VRAM.

Dataseturile mari produc câștiguri NLL spectaculoase pe testul generat din
aceleași familii de șabloane. Controlul pe snapshotul seed, scris înaintea
generatorului și absent din trainingul nou, reduce însă câștigurile la valori
mici pentru Qwen și produce regresii pentru Gemma și Llama. Concluzia onestă
este:

- mecanismul MI este validat mai solid;
- scalarea sintetică a datasetului nu validează încă generalizarea;
- Qwen este singurul candidat rezonabil pentru următoarea iterație;
- Gemma trebuie respins temporar pentru acest setup;
- Llama necesită investigarea severă a latenței de decodare cu blocuri injectate;
- LSR are top‑1 perfect în distribuție, dar numai 82,76% pe snapshotul vechi,
  iar politica `top_k=3` selectează prea multe fațete pentru un context agentic.

Acesta este un rezultat bun pentru o lucrare de licență: există atât un mecanism
funcțional, cât și un eșec măsurat care justifică următoarele experimente.

## Protocol

Hardware și runtime:

- NVIDIA GeForce RTX 5070 Laptop GPU, 8.151 MiB;
- Python 3.11.15;
- PyTorch 2.11.0 + CUDA 12.8;
- BF16 și gradient checkpointing;
- checkpointuri locale, fără download în timpul benchmarkului.

Modele:

| Familie | Checkpoint | Parametri bază |
|---|---|---:|
| Qwen | Qwen2.5-1.5B-Instruct | 1.543.714.304 |
| Gemma | Gemma-3-1B-IT | 999.885.952 |
| Llama | Llama-3.2-1B-Instruct | 1.235.814.400 |

Topologie MI:

```text
base
base + coder                         # două blocuri, două sloturi
base + coder + coder.python          # plus un bloc copil
```

Pozițiile au fost păstrate comparabile cu smoke benchmarkul și adaptate la
numărul de straturi al fiecărei familii. Copilul a fost antrenat cu părintele
activ și înghețat.

Training per fațetă:

- 2.000 continued-pretraining + 2 x 2.500 supervised = 7.000 exemple logice;
- 256 optimizer steps, batch 1, secvență maximă 512;
- bugetul acoperă aproximativ `0,0366` epoci nominale; este un benchmark de
  integrare și răspuns timpuriu, nu antrenare până la convergență pe toate rândurile;
- AdamW, cosine schedule, warmup 5%, clipping la 1.0;
- LR Qwen/Llama `1e-5`;
- LR Gemma `5e-7` părinte și `1e-6` copil, din cauza gradientelor foarte mari;
- o singură sămânță (`42`), deci nu există încă intervale de încredere între runuri.

Evaluarea principală folosește câte 128 de exemple per fațetă, eșantionate
determinist și proporțional după limbă și context din cele 2.000 de teste
sintetice candidate.
Niciun exemplu test nu intră în training. Generarea folosește câte un caz per
fațetă și stare, limitat la 128 tokeni.

Controlul cross-template folosește toate cele șase exemple SFT inițiale per
fațetă din `seed-snapshots/initial-2026-08-23`. Ele nu există în corpusul nou și
folosesc contractele v2, în timp ce corpusul mare folosește v3.

## Training și costul artefactelor

| Familie | Fațetă | Parametri antrenați | Artefact | Timp | Peak CUDA train | Mean loss | Last loss |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | `coder` | 93,60 M | 178,52 MiB | 112,77 s | 4.734 MiB | 2,2774 | 1,0353 |
| Qwen | `coder.python` | 46,80 M | 89,26 MiB | 63,30 s | 4.679 MiB | 1,6855 | 0,8320 |
| Gemma | `coder` | 53,68 M | 102,40 MiB | 92,79 s | 4.076 MiB | 0,7204 | 0,0273 |
| Gemma | `coder.python` | 26,84 M | 51,20 MiB | 57,23 s | 4.213 MiB | 0,5022 | 0,0538 |
| Llama | `coder` | 121,64 M | 232,02 MiB | 66,71 s | 4.146 MiB | 1,4655 | 0,4620 |
| Llama | `coder.python` | 60,82 M | 116,01 MiB | 52,81 s | 4.115 MiB | 1,3678 | 0,7957 |

Toate cele șase identity-check-uri:

```text
maxAbsoluteLogitDifference = 0.0
meanAbsoluteLogitDifference = 0.0
passed = true
```

Gemma a avut frecvent norme brute ale gradientului de ordinul sutelor sau
miilor înainte de clipping. Loss-ul de train extrem de mic nu este un avantaj
de încredere; rezultatul cross-template arată overfitting și/sau sensibilitate
la scalare.

## NLL pe testul generat din aceleași familii

Valori mai mici sunt mai bune.

### Contractul `coder`

| Familie | Base | Base + coder | Stack complet | Delta părinte/base | Delta stack/base | Interferență copil vs părinte |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 3,2617 | 1,0082 | 1,0371 | −69,09% | −68,20% | +2,87% |
| Gemma | 4,6663 | 0,0971 | 0,6189 | −97,92% | −86,74% | +537,51% |
| Llama | 3,8799 | 0,5309 | 0,6114 | −86,32% | −84,24% | +15,16% |

### Contractul `coder.python`

| Familie | Base | Base + coder | Stack complet | Delta părinte/base | Delta stack/base | Delta copil vs părinte |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 1,2905 | 1,2289 | 0,7021 | −4,77% | −45,59% | −42,87% |
| Gemma | 2,3369 | 2,8979 | 0,1851 | +24,01% | −92,08% | −93,61% |
| Llama | 1,9791 | 2,2011 | 0,6846 | +11,22% | −65,41% | −68,90% |

Testul arată că fațetele pot învăța contractul și că un copil poate adăuga o
competență separată. Arată și interferență: copilul degradează contractul
părintelui la toate familiile, în special Gemma, iar părintele singur degradează
Python la Gemma și Llama.

Aceste cifre nu trebuie prezentate ca generalizare, deoarece generatorul de
train și cel de test împart aceleași 32 de familii coder/Python și aceeași
gramatică de răspuns.

## Control cross-template pe snapshotul inițial

### `coder` v2, șase cazuri per familie

| Familie | Base | Base + coder | Stack complet | Delta părinte/base | Delta stack/base |
|---|---:|---:|---:|---:|---:|
| Qwen | 3,0841 | 2,9978 | 2,9612 | −2,80% | −3,98% |
| Gemma | 4,3890 | 6,0994 | 6,6519 | +38,97% | +51,56% |
| Llama | 3,6804 | 3,7065 | 3,7739 | +0,71% | +2,54% |

### `coder.python` v2, șase cazuri per familie

| Familie | Base | Base + coder | Stack complet | Delta părinte/base | Delta stack/base |
|---|---:|---:|---:|---:|---:|
| Qwen | 1,6420 | 1,6014 | 1,4970 | −2,47% | −8,83% |
| Gemma | 3,1353 | 3,8910 | 3,4850 | +24,10% | +11,15% |
| Llama | 2,3101 | 2,4743 | 2,2351 | +7,11% | −3,24% |

Acesta este cel mai important rezultat al reevaluării. Qwen păstrează un câștig
mic în afara generatorului. Gemma inversează câștigul într-o regresie severă.
Llama este practic neutru pe `coder` și are un câștig mic doar cu copilul pe
Python. Eșantionul are numai șase cazuri, deci este un detector de risc, nu un
benchmark suficient; tocmai de aceea următorul dataset prioritar trebuie să fie
un gold set independent.

## Memorie și generare

Peak-ul PyTorch în evaluare:

| Familie | Base | Base + coder | Stack complet | Parametri activi în stack |
|---|---:|---:|---:|---:|
| Qwen | 5.930 MiB | 6.109 MiB | 6.198 MiB | 140,39 M |
| Gemma | 7.094 MiB | 7.201 MiB | 7.254 MiB | 80,53 M |
| Llama | 4.888 MiB | 5.120 MiB | 5.236 MiB | 182,46 M |

Toate stările au încăput pe 8 GB, dar telemetria `nvidia-smi` a ajuns la
aproximativ 7,7 GiB în stivele cele mai grele. Headroom-ul pentru context mai
lung, batch mai mare sau alt proces GPU este mic.

Throughput generativ observat, agregat pe cele două prompturi/stare:

| Familie | Base | Base + coder | Stack complet |
|---|---:|---:|---:|
| Qwen | 35,66 tok/s | 37,05 tok/s | 36,07 tok/s |
| Gemma | 34,29 tok/s | 33,71 tok/s | 32,50 tok/s |
| Llama | 59,14 tok/s | 4,11 tok/s | 3,82 tok/s |

Llama are o regresie de aproximativ 15x la decodare când sunt injectate
blocuri. Acest lucru indică o problemă de cache/poziționare sau o cale de
execuție ineficientă specifică arhitecturii și trebuie investigat înainte ca
Llama să fie candidat agentic.

Niciun output nu a închis un obiect JSON-rădăcină complet în bugetul de 128
tokeni. Scorerul inițial accepta greșit obiecte interne complete dintr-un root
trunchiat; a fost corectat la `root-json-v2`, iar outputurile au fost rescored
fără rerularea modelului. Rata de menționare a câmpurilor crește în general cu
fațetele, dar testul generativ este prea scurt pentru a valida contractele.

## LSR pe aceleași rânduri canonice

Pentru fiecare dintre cele cinci fațete a fost construită o semnătură v3 din
exact cele 4.500 rânduri de pretraining+training. Fiecare semnătură are:

```text
L1 = 128
L2 = 1.280
L3 = 4.500
dimensiune pe disc ~= 8,69 MiB
```

Testul folosește toate cele 10.000 de rânduri `test.jsonl`, modelul
`all-MiniLM-L6-v2`, RTX 5070 pentru embeddings, index policy implicită și
`top_k=3`.

### Test generat in-distribution, 10.000 de cazuri

| Metrică | Rezultat |
|---|---:|
| top‑1 recall | 100,00% |
| top‑3 recall | 100,00% |
| MRR | 1,0000 |
| false-positive top‑1 | 0,00% |
| precision pe setul selectat | 33,33% |
| recall pe setul selectat | 100,00% |
| F1 pe setul selectat | 50,00% |
| exact-set | 0,00% |
| over-selection | 100,00% |
| fațete suplimentare median / p95 | 2 / 2 |
| latență p50 / p95 / p99 | 72,70 / 77,34 / 86,00 ms |

Primul candidat este corect în toate cele 10.000 de cazuri, dar configurația
`top_k=3`, fără praguri explicite, întoarce invariabil trei fațete. Față de
eticheta directă unică, două sunt deci suplimentare la fiecare cerere. Acesta
este în primul rând un eșec al politicii de selecție configurate, nu al ordinii
top‑1: pentru operatorii temporali exclusivi, runtime-ul trebuie să folosească
`top_k=1` sau praguri/margini calibrate; MI rezolvă apoi părinții necesari.

### Control cross-template pe snapshotul inițial, 29 de cazuri

Aceleași semnături au fost evaluate și pe testele mici scrise înaintea
generatorului nou și excluse din corpusurile de 4.500 de rânduri:

| Metrică | Rezultat |
|---|---:|
| top‑1 / exact-set la `top_k=1` | 82,76% |
| false-positive top‑1 | 17,24% |
| top‑3 recall | 96,55% |
| MRR la `top_k=3` | 0,8908 |

| Fațetă așteptată | Corecte top‑1 | Rată |
|---|---:|---:|
| `coder` | 6/6 | 100,00% |
| `coder.python` | 4/6 | 66,67% |
| `planner` | 6/6 | 100,00% |
| `tooler` | 5/5 | 100,00% |
| `verifier` | 3/6 | 50,00% |

Patru dintre cele cinci erori top‑1 rămân în top‑3; un caz nu este recuperat
nici acolo. Slăbiciunile sunt concentrate în frontiera părinte–copil
`coder`/`coder.python` și în `verifier`. Eșantionul de 29 este prea mic pentru o
estimare stabilă, dar demontează scorul aparent perfect din distribuția
generatorului și indică exact unde trebuie construite hard negatives.

Interpretarea corectă trebuie să țină cont de două limite:

1. train/test împart generatorul și vocabularul contractelor, deci scorul top‑1
   măsoară în principal separabilitatea șabloanelor;
2. `top_k=3` împreună cu pragurile actuale poate activa operatori temporali
   incompatibili simultan, chiar dacă primul rezultat este corect.

Evaluatorul a fost extins cu precision/recall/F1 pe set, exact-set,
over-selection, under-selection și numărul de fațete suplimentare. Eticheta
canonică a fost corectată să folosească selecția directă (`primaryFacet` sau
`directSelectedFacets`), nu closure-ul `activeFacets`; rezolvarea părinților
aparține MI.

## Validarea dataseturilor

- 15/15 JSONL-uri au 2.000–2.500 rânduri;
- 32.500 exemple canonice, aproximativ 108,4 MiB;
- 14 limbi eșantionate și 20 contexte echilibrate;
- zero ID-uri, prompturi normalizate sau semantic cases duplicate între splituri;
- toate targeturile contractuale sunt JSON parseabil;
- 9.000 fișiere Python trec AST;
- 4.500/4.500 implementări Python au trecut câte un test executat;
- validatorul infrastructurii raportează 19 manifeste, 32.512 recorduri cu
  benchmarkurile agentice și zero avertismente;
- exportul LSR raportează 4.500 route texts per fațetă și 10.000 teste;
- 11/11 teste MI și 27/27 teste LSR trec.

## Decizie pe familie

### Qwen — candidat principal

Este singurul model cu îmbunătățire în ambele controale cross-template și fără
penalizare generativă evidentă. Nu este încă „câștigător”: rezultatul extern are
numai șase cazuri, iar stack-ul adaugă 140,39 M parametri.

### Gemma — respins temporar

Are cel mai bun fit pe testul generat și cel mai rău transfer. Gradientele brute
foarte mari, regresia `coder` de +51,56% și regresia Python de +11,15% pe seed
fac setup-ul actual nesigur. Necesită LR/normalization study, mai multe seed-uri
și loss de conservare înainte de altă comparație.

### Llama — inconcludent calitativ, respins operațional în runtime-ul actual

Transferul este aproape neutru, iar latența generativă se degradează de circa
15x. Poate rămâne control arhitectural, dar nu candidat principal pentru edge
agentic până la rezolvarea cache-ului.

## Ce trebuie schimbat înainte de următorul training mare

1. Îngheață corpusurile actuale ca `synthetic-bootstrap-v1`; nu le promova la
   `trainable` sau `accepted`.
2. Construiește 200–500 exemple gold independente per fațetă, cu taskuri și
   autori diferiți de generator; ține acest set complet în afara trainingului.
3. Fă split pe familii de problemă și template, nu doar pe `semanticCaseId`.
4. Pentru copil, adaugă replay din părintele înghețat și/sau KL/retention loss;
   măsoară explicit regresia părintelui la fiecare checkpoint.
5. Rulează Qwen cu minimum trei seed-uri și ablații 1 vs 2 sloturi pentru
   părinte, apoi 1 bloc copil; raportează medie și dispersie.
6. Separă rutarea LSR în trei gate-uri: un operator rădăcină exclusiv sau
   abstain, specializări condiționate de părinte și knowledge packs multi-label
   independenți. Closure-ul se rezolvă după selecție.
7. Adaugă hard negatives agentice și cazuri `no-facet`; calibrează pragurile pe
   validation și raportează precision/exact-match, nu doar recall@k.
8. Înlocuiește cadrele code-switch cu exemple native scrise/revizuite pentru
   fiecare limbă și raportează metrici per limbă.
9. Evaluează `coder.python` pe probleme executabile nevăzute și `coder` cu
   rubrici independente; NLL pe JSON sintetic nu este suficient.
10. Optimizează evaluatorul LSR prin batching și progres, fără a schimba
    comportamentul online per-request.

## Reproducere și artefacte

Configurațiile sunt în `MI/benchmarks/model-family-2k/configs`. Runnerul comun:

```powershell
python MI\benchmarks\model-family-smoke\run_benchmark.py --profile data2k
python MI\benchmarks\model-family-smoke\run_benchmark.py --profile data2k --seed-holdout --skip-training
python MI\benchmarks\model-family-smoke\run_benchmark.py --profile data2k --rescore-only
```

Rezultate machine-readable:

- `results/qwen.json`, `gemma.json`, `llama.json`;
- `results/summary.json`;
- `results-seed-holdout/*.json`;
- `results/lsr-routing.json`;
- `results/lsr-routing-seed-holdout-top1.json` și
  `results/lsr-routing-seed-holdout-top3.json`;
- `results/lsr-routing-seed-*-top1.json`, pentru defalcarea pe fațetă.

Fațetele antrenate sunt păstrate separat de smoke în
`facets/mi-benchmark-2k/<family>/<facet>/data2k-2026-08-23`. Semnăturile LSR
sunt în `facets/signatures-data2k`, fără a suprascrie registry-ul anterior.
