# Raport MI: Qwen, Gemma și Llama pe RTX 5070

Data rulării: 23 august 2026. Acesta este un benchmark de fezabilitate pentru
Componenta 2/4 (Mixed Inference), nu o evaluare suficientă pentru alegerea
definitivă a modelului de bază și nu o demonstrație de calitate competitivă.

## Verdict scurt

Mecanismul MI funcționează end-to-end pe toate cele trei arhitecturi: modelele
se încarcă offline, blocurile se inserează reversibil, inițializarea este
identică la nivel de logits, părintele rămâne înghețat și activ la trainingul
copilului, artefactele se salvează și copilul rezolvă automat dependența.

Rezultatul de calitate este însă mixt. Qwen și Llama reduc NLL held-out în mod
monoton. Gemma reduce puternic NLL pe Python după reglarea learning rate-ului,
dar copilul produce regresie pe competența generică `coder`. Metricile de
generare nu sunt stabile pe numai două exemple held-out per fațetă. Concluzia
corectă este „MI este executabil și măsurabil”, nu „fațetele sunt deja bune”.

## Hardware și mediu

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8.151 MiB raportați de driver;
- CPU: AMD Ryzen AI 9 365, 10 nuclee / 20 fire;
- Python 3.11.15, PyTorch 2.11.0+cu128;
- training și inferență în BF16 pe `cuda:0`;
- `torch.cuda.max_memory_allocated` măsoară numai memoria alocată de PyTorch,
  nu memoria desktopului, memoria rezervată de allocator sau consumul total al
  plăcii;
- NPU-ul Ryzen nu a fost folosit: MI v1 are backend PyTorch single-device, iar
  mediul instalat nu expune un backend NPU AMD compatibil.

Toate rulările au încăput pe RTX 5070. Vârful PyTorch la training a fost 4,012
GiB, pentru părintele Qwen.

## Modelele locale

| Familie | Parametri bază | Straturi | Checkpoint BF16 | Sursă locală | Situație licență |
|---|---:|---:|---:|---|---|
| Qwen2.5 1.5B Instruct | 1.543.714.304 | 28 | 2.944,44 MiB | oficial `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 |
| Gemma 3 1B IT | 999.885.952 | 26 | 1.907,17 MiB | oglindă publică `unsloth/gemma-3-1b-it` a checkpointului Google | Gemma Terms of Use |
| Llama 3.2 1B Instruct | 1.235.814.400 | 16 | 2.357,14 MiB | oglindă publică `unsloth/Llama-3.2-1B-Instruct` a checkpointului Meta | Llama 3.2 Community License |

Depozitele oficiale Google și Meta sunt gated și cer autentificare plus
acceptarea personală a licenței. Nu a fost acceptat un acord în numele
operatorului. Proveniența, reviziile Hub și hashurile SHA-256 complete sunt în
`base-models/downloaded/README.md` și `models.json`. Auditul local a găsit zero
fișiere incomplete sau lock-uri rămase.

## Topologia aleasă

Topologia este normalizată relativ la adâncimea decoderului:

| Familie | `coder` — două sloturi / două blocuri | `coder.python` — un slot / un bloc | Blocuri în stiva finală |
|---|---|---|---:|
| Qwen | după straturile 8 și 18 | după stratul 23 | 3 |
| Gemma | după straturile 8 și 16 | după stratul 21 | 3 |
| Llama | după straturile 4 și 9 | după stratul 12 | 3 |

Fiecare bloc folosește `identity-copy` din stratul precedent. `coder.python`
declară `requires: [coder@smoke-2026-08-23]`; la training părintele este activ
și înghețat, iar la generate solicitarea copilului rezolvă arborele complet.

## Protocolul de date și training

Pentru fiecare fațetă:

- 10 exemple `continued-pretraining`;
- primele 4 din cele 6 exemple SFT pentru training;
- ultimele 2 exemple SFT păstrate în afara trainingului pentru evaluare;
- 6 pași AdamW, batch 1, gradient accumulation 1, gradient checkpointing;
- secvențe de maximum 256 tokeni la training;
- seed 42 și gradient clipping la 1,0;
- `5e-5` pentru Qwen/Llama, `2e-6` pentru Gemma.

Fișierele canonice `test.jsonl` nu conțin `target.messages`, deci nu permit
NLL sau comparație cu un răspuns gold. Nu au fost inventate etichete pentru
ele. Cele două exemple SFT reținute per fațetă oferă un holdout real, dar foarte
mic. Evaluarea folosește maximum 512 tokeni și nu a trunchiat niciun record.
Generarea deterministă folosește chat template-ul modelului și maximum 256
tokeni noi.

## Rezultatul trainingului

Toate cele șase verificări `identity-copy` au raportat diferență maximă de
logits `0,0` înainte de primul update.

| Familie / fațetă | Mean train loss | Last train loss | Timp | Peak CUDA allocated | Artefact |
|---|---:|---:|---:|---:|---:|
| Qwen / `coder` | 4,1185 | 2,1269 | 7,727 s | 4,012 GiB | 178,52 MiB |
| Qwen / `coder.python` | 3,8412 | 0,7813 | 3,308 s | 3,942 GiB | 89,26 MiB |
| Gemma / `coder` | 4,2985 | 2,9897 | 7,716 s | 3,194 GiB | 102,40 MiB |
| Gemma / `coder.python` | 4,2301 | 1,4351 | 3,944 s | 3,151 GiB | 51,20 MiB |
| Llama / `coder` | 4,0913 | 2,5185 | 7,053 s | 3,523 GiB | 232,02 MiB |
| Llama / `coder.python` | 3,6550 | 1,2530 | 3,562 s | 3,431 GiB | 116,01 MiB |

Lossurile de training provin din exemple diferite, amestecate și foarte puține;
nu trebuie comparate ca un clasament între modele.

### Trialul Gemma respins

Rata comună `5e-5` a trecut identitatea, dar a produs normă de gradient până la
9.536 și degradare catastrofică held-out după numai șase pași: `coder` 4,219 →
8,272 cu părintele, iar `python` 3,658 → 6,794. Trialul este păstrat în
`gemma-unstable-trial-summary.json`. Rularea finală folosește `2e-6`, fără
schimbarea datelor sau a topologiei. Din acest motiv comparația între familii
nu este un experiment strict cu hiperparametri identici.

## Costul structural

| Familie | Parametri `base+coder` | Overhead | Parametri toate fațetele | Overhead | Total artefacte |
|---|---:|---:|---:|---:|---:|
| Qwen | 93.595.648 | 6,063% | 140.393.472 | 9,095% | 267,78 MiB |
| Gemma | 53.684.224 | 5,369% | 80.526.336 | 8,054% | 153,60 MiB |
| Llama | 121.643.008 | 9,843% | 182.464.512 | 14,765% | 348,03 MiB |

Un „bloc” nu are același cost între familii. Llama are numai 16 straturi, dar
fiecare bloc este mai mare; trei blocuri adaugă aproape 15% din parametrii bazei.
Numărul de sloturi nu poate fi ales izolat de dimensiunea blocului.

## NLL held-out

Mai mic este mai bun. Procentele sunt față de baza aceleiași familii.

| Familie | Stare | `coder` NLL | Δ | `coder.python` NLL | Δ |
|---|---|---:|---:|---:|---:|
| Qwen | base | 3,3889 | — | 1,9938 | — |
| Qwen | base+facet | 3,3149 | -2,18% | 1,9506 | -2,17% |
| Qwen | base+parent+child | 3,2879 | -2,98% | 1,9206 | -3,67% |
| Gemma | base | 4,2185 | — | 3,6576 | — |
| Gemma | base+facet | 4,0976 | -2,87% | 3,2552 | -11,00% |
| Gemma | base+parent+child | 4,5277 | **+7,33%** | 3,2179 | -12,02% |
| Llama | base | 3,8523 | — | 2,7625 | — |
| Llama | base+facet | 3,6370 | -5,59% | 2,7310 | -1,14% |
| Llama | base+parent+child | 3,6216 | -5,99% | 2,6818 | -2,92% |

Qwen și Llama au evoluție monotonă pe ambele holdouturi. Gemma arată cea mai
mare specializare Python, dar și cea mai clară uitare/interferență asupra
părintelui după activarea copilului.

## Generate, viteză și memorie

`Strict` este media componentelor: JSON valid, acoperirea câmpurilor în JSON și,
pentru copil, cod Python parsabil. `Structure` înlocuiește acoperirea JSON cu
menționarea câmpurilor și în text/Markdown. Fiecare valoare este media a numai
patru generații, deci este diagnostic, nu benchmark statistic.

| Familie | Stare | tokeni/s | Peak CUDA | Strict | Structure | F1 lexical |
|---|---|---:|---:|---:|---:|---:|
| Qwen | base | 39,41 | 3,327 GiB | 0,083 | 0,244 | 0,150 |
| Qwen | base+facet | 36,87 | 3,501 GiB | 0,250 | 0,375 | 0,122 |
| Qwen | base+parent+child | 36,07 | 3,588 GiB | 0,042 | 0,185 | 0,137 |
| Gemma | base | 37,15 | 2,718 GiB | 0,000 | 0,071 | 0,119 |
| Gemma | base+facet | 34,85 | 2,822 GiB | 0,083 | 0,119 | 0,065 |
| Gemma | base+parent+child | 33,43 | 2,874 GiB | 0,083 | 0,226 | 0,052 |
| Llama | base | 61,53 | 2,722 GiB | 0,083 | 0,083 | 0,132 |
| Llama | base+facet | 56,74 | 2,948 GiB | 0,000 | 0,028 | 0,090 |
| Llama | base+parent+child | 53,64 | 3,061 GiB | 0,062 | 0,062 | 0,096 |

Stiva completă reduce throughputul față de bază cu aproximativ 8,5% la Qwen,
10,0% la Gemma și 12,8% la Llama. Llama este cel mai rapid în această rulare,
dar plătește cel mai mare overhead procentual în parametri și memorie pentru
cele trei blocuri.

## Probleme identificate

1. Două exemple held-out per fațetă nu pot estima generalizarea sau variația.
2. `test.jsonl` este potrivit pentru rutare și rubrici, dar nu are răspuns gold;
   trebuie completat dacă vrem evaluare automată de calitate din aceeași sursă.
3. Trainingul copilului nu include o penalizare de conservare a părintelui;
   regresia Gemma pe `coder` arată că simplul „parent frozen + active” nu este
   suficient pentru compoziție fără interferență.
4. Learning rate-ul nu este transferabil între familii. Gemma necesită tuning
   sau o normalizare mai bună a update-urilor blocurilor.
5. Trei blocuri costă 8–15% din bază. O bibliotecă mare de fațete complete va
   deveni rapid scumpă pe disc și în VRAM, chiar dacă baza este partajată.
6. Scăderea NLL nu garantează respectarea contractului sau cod corect. Scorurile
   de structură și F1 sunt volatile și uneori se mișcă în direcția opusă.
7. Benchmarkul validează RTX/CUDA, nu Ryzen NPU, cuantizare, offload, batching
   concurent sau traseul agentic LSR → MI.

## Direcția recomandată

Înainte de mii de exemple per fațetă:

1. completați 50–100 exemple gold de evaluare per fațetă, fără a le folosi la
   training sau la embeddings de selecție a hiperparametrilor;
2. adăugați la copil replay din părinți și o pierdere de conservare/KL față de
   `base+parent`, apoi măsurați explicit regresia pe fiecare strămoș;
3. faceți ablații 1/2/3 sloturi și 1/2 blocuri, raportând câștigul per milion de
   parametri și per GiB, nu numai lossul final;
4. folosiți learning-rate range tests per arhitectură și oprire timpurie pe un
   validation split separat; Gemma nu trebuie antrenată cu presetul Qwen;
5. abia după stabilizarea unei topologii extindeți `coder` și `coder.python` la
   2.000–5.000 exemple, cu replay multilingv și cazuri negative;
6. evaluați ulterior pe seturi externe de coding plus teste executabile, nu doar
   similaritate lexicală;
7. păstrați Llama ca reper de viteză, Qwen ca reper de stabilitate/compoziție și
   Gemma ca test de stres pentru interferență; nu alegeți încă un singur câștigător.

## Artefacte și reproducere

- modele: `base-models/downloaded/<familie>/<checkpoint>`;
- fațete: `facets/mi-benchmark/<familie>/<facet>/smoke-2026-08-23`;
- configurații: `MI/benchmarks/model-family-smoke/configs`;
- runner: `MI/benchmarks/model-family-smoke/run_benchmark.py`;
- rezultate complete și generații: `results/qwen.json`, `gemma.json`,
  `llama.json` și `summary.json`.

Comanda completă este documentată în README-ul benchmarkului. Au trecut 11/11
teste MI, inclusiv training/resume/generate end-to-end pe un Llama miniatural.
Fiecare dintre cele șase artefacte reale a fost reîncărcat și folosit în
evaluarea compusă; hashurile manifestelor și ale fișierelor safetensors au fost
validate la încărcare.
