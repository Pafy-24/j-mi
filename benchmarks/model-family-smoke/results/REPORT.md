# Raport MI: Qwen, Gemma Č™i Llama pe RTX 5070

Data rulÄrii: 23 august 2026. Acesta este un benchmark de fezabilitate pentru
Componenta 2/5 (Mixed Inference), nu o evaluare suficientÄ pentru alegerea
definitivÄ a modelului de bazÄ Č™i nu o demonstraČ›ie de calitate competitivÄ.

## Verdict scurt

Mecanismul MI funcČ›ioneazÄ end-to-end pe toate cele trei arhitecturi: modelele
se Ă®ncarcÄ offline, blocurile se insereazÄ reversibil, iniČ›ializarea este
identicÄ la nivel de logits, pÄrintele rÄmĂ˘ne Ă®ngheČ›at Č™i activ la trainingul
copilului, artefactele se salveazÄ Č™i copilul rezolvÄ automat dependenČ›a.

Rezultatul de calitate este Ă®nsÄ mixt. Qwen Č™i Llama reduc NLL held-out Ă®n mod
monoton. Gemma reduce puternic NLL pe Python dupÄ reglarea learning rate-ului,
dar copilul produce regresie pe competenČ›a genericÄ `coder`. Metricile de
generare nu sunt stabile pe numai douÄ exemple held-out per faČ›etÄ. Concluzia
corectÄ este â€žMI este executabil Č™i mÄsurabilâ€ť, nu â€žfaČ›etele sunt deja buneâ€ť.

## Hardware Č™i mediu

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8.151 MiB raportaČ›i de driver;
- CPU: AMD Ryzen AI 9 365, 10 nuclee / 20 fire;
- Python 3.11.15, PyTorch 2.11.0+cu128;
- training Č™i inferenČ›Ä Ă®n BF16 pe `cuda:0`;
- `torch.cuda.max_memory_allocated` mÄsoarÄ numai memoria alocatÄ de PyTorch,
  nu memoria desktopului, memoria rezervatÄ de allocator sau consumul total al
  plÄcii;
- NPU-ul Ryzen nu a fost folosit: MI v1 are backend PyTorch single-device, iar
  mediul instalat nu expune un backend NPU AMD compatibil.

Toate rulÄrile au Ă®ncÄput pe RTX 5070. VĂ˘rful PyTorch la training a fost 4,012
GiB, pentru pÄrintele Qwen.

## Modelele locale

| Familie | Parametri bazÄ | Straturi | Checkpoint BF16 | SursÄ localÄ | SituaČ›ie licenČ›Ä |
|---|---:|---:|---:|---|---|
| Qwen2.5 1.5B Instruct | 1.543.714.304 | 28 | 2.944,44 MiB | oficial `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 |
| Gemma 3 1B IT | 999.885.952 | 26 | 1.907,17 MiB | oglindÄ publicÄ `unsloth/gemma-3-1b-it` a checkpointului Google | Gemma Terms of Use |
| Llama 3.2 1B Instruct | 1.235.814.400 | 16 | 2.357,14 MiB | oglindÄ publicÄ `unsloth/Llama-3.2-1B-Instruct` a checkpointului Meta | Llama 3.2 Community License |

Depozitele oficiale Google Č™i Meta sunt gated Č™i cer autentificare plus
acceptarea personalÄ a licenČ›ei. Nu a fost acceptat un acord Ă®n numele
operatorului. ProvenienČ›a, reviziile Hub Č™i hashurile SHA-256 complete sunt Ă®n
`base-models/downloaded/README.md` Č™i `models.json`. Auditul local a gÄsit zero
fiČ™iere incomplete sau lock-uri rÄmase.

## Topologia aleasÄ

Topologia este normalizatÄ relativ la adĂ˘ncimea decoderului:

| Familie | `coder` â€” douÄ sloturi / douÄ blocuri | `coder.python` â€” un slot / un bloc | Blocuri Ă®n stiva finalÄ |
|---|---|---|---:|
| Qwen | dupÄ straturile 8 Č™i 18 | dupÄ stratul 23 | 3 |
| Gemma | dupÄ straturile 8 Č™i 16 | dupÄ stratul 21 | 3 |
| Llama | dupÄ straturile 4 Č™i 9 | dupÄ stratul 12 | 3 |

Fiecare bloc foloseČ™te `identity-copy` din stratul precedent. `coder.python`
declarÄ `requires: [coder@smoke-2026-08-23]`; la training pÄrintele este activ
Č™i Ă®ngheČ›at, iar la generate solicitarea copilului rezolvÄ arborele complet.

## Protocolul de date Č™i training

Pentru fiecare faČ›etÄ:

- 10 exemple `continued-pretraining`;
- primele 4 din cele 6 exemple SFT pentru training;
- ultimele 2 exemple SFT pÄstrate Ă®n afara trainingului pentru evaluare;
- 6 paČ™i AdamW, batch 1, gradient accumulation 1, gradient checkpointing;
- secvenČ›e de maximum 256 tokeni la training;
- seed 42 Č™i gradient clipping la 1,0;
- `5e-5` pentru Qwen/Llama, `2e-6` pentru Gemma.

FiČ™ierele canonice `test.jsonl` nu conČ›in `target.messages`, deci nu permit
NLL sau comparaČ›ie cu un rÄspuns gold. Nu au fost inventate etichete pentru
ele. Cele douÄ exemple SFT reČ›inute per faČ›etÄ oferÄ un holdout real, dar foarte
mic. Evaluarea foloseČ™te maximum 512 tokeni Č™i nu a trunchiat niciun record.
Generarea deterministÄ foloseČ™te chat template-ul modelului Č™i maximum 256
tokeni noi.

## Rezultatul trainingului

Toate cele Č™ase verificÄri `identity-copy` au raportat diferenČ›Ä maximÄ de
logits `0,0` Ă®nainte de primul update.

| Familie / faČ›etÄ | Mean train loss | Last train loss | Timp | Peak CUDA allocated | Artefact |
|---|---:|---:|---:|---:|---:|
| Qwen / `coder` | 4,1185 | 2,1269 | 7,727 s | 4,012 GiB | 178,52 MiB |
| Qwen / `coder.python` | 3,8412 | 0,7813 | 3,308 s | 3,942 GiB | 89,26 MiB |
| Gemma / `coder` | 4,2985 | 2,9897 | 7,716 s | 3,194 GiB | 102,40 MiB |
| Gemma / `coder.python` | 4,2301 | 1,4351 | 3,944 s | 3,151 GiB | 51,20 MiB |
| Llama / `coder` | 4,0913 | 2,5185 | 7,053 s | 3,523 GiB | 232,02 MiB |
| Llama / `coder.python` | 3,6550 | 1,2530 | 3,562 s | 3,431 GiB | 116,01 MiB |

Lossurile de training provin din exemple diferite, amestecate Č™i foarte puČ›ine;
nu trebuie comparate ca un clasament Ă®ntre modele.

### Trialul Gemma respins

Rata comunÄ `5e-5` a trecut identitatea, dar a produs normÄ de gradient pĂ˘nÄ la
9.536 Č™i degradare catastroficÄ held-out dupÄ numai Č™ase paČ™i: `coder` 4,219 â†’
8,272 cu pÄrintele, iar `python` 3,658 â†’ 6,794. Trialul este pÄstrat Ă®n
`gemma-unstable-trial-summary.json`. Rularea finalÄ foloseČ™te `2e-6`, fÄrÄ
schimbarea datelor sau a topologiei. Din acest motiv comparaČ›ia Ă®ntre familii
nu este un experiment strict cu hiperparametri identici.

## Costul structural

| Familie | Parametri `base+coder` | Overhead | Parametri toate faČ›etele | Overhead | Total artefacte |
|---|---:|---:|---:|---:|---:|
| Qwen | 93.595.648 | 6,063% | 140.393.472 | 9,095% | 267,78 MiB |
| Gemma | 53.684.224 | 5,369% | 80.526.336 | 8,054% | 153,60 MiB |
| Llama | 121.643.008 | 9,843% | 182.464.512 | 14,765% | 348,03 MiB |

Un â€žblocâ€ť nu are acelaČ™i cost Ă®ntre familii. Llama are numai 16 straturi, dar
fiecare bloc este mai mare; trei blocuri adaugÄ aproape 15% din parametrii bazei.
NumÄrul de sloturi nu poate fi ales izolat de dimensiunea blocului.

## NLL held-out

Mai mic este mai bun. Procentele sunt faČ›Ä de baza aceleiaČ™i familii.

| Familie | Stare | `coder` NLL | Î” | `coder.python` NLL | Î” |
|---|---|---:|---:|---:|---:|
| Qwen | base | 3,3889 | â€” | 1,9938 | â€” |
| Qwen | base+facet | 3,3149 | -2,18% | 1,9506 | -2,17% |
| Qwen | base+parent+child | 3,2879 | -2,98% | 1,9206 | -3,67% |
| Gemma | base | 4,2185 | â€” | 3,6576 | â€” |
| Gemma | base+facet | 4,0976 | -2,87% | 3,2552 | -11,00% |
| Gemma | base+parent+child | 4,5277 | **+7,33%** | 3,2179 | -12,02% |
| Llama | base | 3,8523 | â€” | 2,7625 | â€” |
| Llama | base+facet | 3,6370 | -5,59% | 2,7310 | -1,14% |
| Llama | base+parent+child | 3,6216 | -5,99% | 2,6818 | -2,92% |

Qwen Č™i Llama au evoluČ›ie monotonÄ pe ambele holdouturi. Gemma aratÄ cea mai
mare specializare Python, dar Č™i cea mai clarÄ uitare/interferenČ›Ä asupra
pÄrintelui dupÄ activarea copilului.

## Generate, vitezÄ Č™i memorie

`Strict` este media componentelor: JSON valid, acoperirea cĂ˘mpurilor Ă®n JSON Č™i,
pentru copil, cod Python parsabil. `Structure` Ă®nlocuieČ™te acoperirea JSON cu
menČ›ionarea cĂ˘mpurilor Č™i Ă®n text/Markdown. Fiecare valoare este media a numai
patru generaČ›ii, deci este diagnostic, nu benchmark statistic.

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

Stiva completÄ reduce throughputul faČ›Ä de bazÄ cu aproximativ 8,5% la Qwen,
10,0% la Gemma Č™i 12,8% la Llama. Llama este cel mai rapid Ă®n aceastÄ rulare,
dar plÄteČ™te cel mai mare overhead procentual Ă®n parametri Č™i memorie pentru
cele trei blocuri.

## Probleme identificate

1. DouÄ exemple held-out per faČ›etÄ nu pot estima generalizarea sau variaČ›ia.
2. `test.jsonl` este potrivit pentru rutare Č™i rubrici, dar nu are rÄspuns gold;
   trebuie completat dacÄ vrem evaluare automatÄ de calitate din aceeaČ™i sursÄ.
3. Trainingul copilului nu include o penalizare de conservare a pÄrintelui;
   regresia Gemma pe `coder` aratÄ cÄ simplul â€žparent frozen + activeâ€ť nu este
   suficient pentru compoziČ›ie fÄrÄ interferenČ›Ä.
4. Learning rate-ul nu este transferabil Ă®ntre familii. Gemma necesitÄ tuning
   sau o normalizare mai bunÄ a update-urilor blocurilor.
5. Trei blocuri costÄ 8â€“15% din bazÄ. O bibliotecÄ mare de faČ›ete complete va
   deveni rapid scumpÄ pe disc Č™i Ă®n VRAM, chiar dacÄ baza este partajatÄ.
6. ScÄderea NLL nu garanteazÄ respectarea contractului sau cod corect. Scorurile
   de structurÄ Č™i F1 sunt volatile Č™i uneori se miČ™cÄ Ă®n direcČ›ia opusÄ.
7. Benchmarkul valideazÄ RTX/CUDA, nu Ryzen NPU, cuantizare, offload, batching
   concurent sau traseul agentic LSR â†’ MI.

## DirecČ›ia recomandatÄ

ĂŽnainte de mii de exemple per faČ›etÄ:

1. completaČ›i 50â€“100 exemple gold de evaluare per faČ›etÄ, fÄrÄ a le folosi la
   training sau la embeddings de selecČ›ie a hiperparametrilor;
2. adÄugaČ›i la copil replay din pÄrinČ›i Č™i o pierdere de conservare/KL faČ›Ä de
   `base+parent`, apoi mÄsuraČ›i explicit regresia pe fiecare strÄmoČ™;
3. faceČ›i ablaČ›ii 1/2/3 sloturi Č™i 1/2 blocuri, raportĂ˘nd cĂ˘Č™tigul per milion de
   parametri Č™i per GiB, nu numai lossul final;
4. folosiČ›i learning-rate range tests per arhitecturÄ Č™i oprire timpurie pe un
   validation split separat; Gemma nu trebuie antrenatÄ cu presetul Qwen;
5. abia dupÄ stabilizarea unei topologii extindeČ›i `coder` Č™i `coder.python` la
   2.000â€“5.000 exemple, cu replay multilingv Č™i cazuri negative;
6. evaluaČ›i ulterior pe seturi externe de coding plus teste executabile, nu doar
   similaritate lexicalÄ;
7. pÄstraČ›i Llama ca reper de vitezÄ, Qwen ca reper de stabilitate/compoziČ›ie Č™i
   Gemma ca test de stres pentru interferenČ›Ä; nu alegeČ›i Ă®ncÄ un singur cĂ˘Č™tigÄtor.

## Artefacte Č™i reproducere

- modele: `base-models/downloaded/<familie>/<checkpoint>`;
- faČ›ete: `facets/mi-benchmark/<familie>/<facet>/smoke-2026-08-23`;
- configuraČ›ii: `MI/benchmarks/model-family-smoke/configs`;
- runner: `MI/benchmarks/model-family-smoke/run_benchmark.py`;
- rezultate complete Č™i generaČ›ii: `results/qwen.json`, `gemma.json`,
  `llama.json` Č™i `summary.json`.

Comanda completÄ este documentatÄ Ă®n README-ul benchmarkului. Au trecut 11/11
teste MI, inclusiv training/resume/generate end-to-end pe un Llama miniatural.
Fiecare dintre cele Č™ase artefacte reale a fost reĂ®ncÄrcat Č™i folosit Ă®n
evaluarea compusÄ; hashurile manifestelor Č™i ale fiČ™ierelor safetensors au fost
validate la Ă®ncÄrcare.
