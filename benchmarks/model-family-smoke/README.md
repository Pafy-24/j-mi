# MI model-family smoke benchmark

Acest benchmark verificÄ end-to-end Componenta 2/5 pe trei arhitecturi mici:
Qwen2, Gemma 3 Č™i Llama 3.2. Nu este un benchmark de calitate generalÄ a
modelelor Č™i nu poate valida specializare realÄ cu numai 14 exemple de
antrenare per faČ›etÄ.

Rezultatul executat Č™i interpretarea criticÄ sunt Ă®n `results/REPORT.md`, iar
valorile Č™i outputurile brute sunt Ă®n `results/summary.json` Č™i fiČ™ierele per
familie.

Pentru fiecare model sunt antrenate Č™i salvate douÄ faČ›ete:

- `coder@smoke-2026-08-23`: douÄ blocuri Ă®n douÄ sloturi, aproximativ la 1/3
  Č™i 2/3 din decoder;
- `coder.python@smoke-2026-08-23`: un bloc la aproximativ 5/6 din decoder,
  dependent de `coder`, care rÄmĂ˘ne activ Č™i Ă®ngheČ›at la trainingul copilului.

ComparaČ›ia are exact trei stÄri:

1. `base`;
2. `base+facet` (`coder`);
3. `base+parent+child` (se cere numai `coder.python`, iar DAG-ul rezolvÄ
   automat pÄrintele).

Primele patru exemple SFT Č™i toate cele zece exemple de continued pretraining
intrÄ Ă®n training. Ultimele douÄ exemple SFT din fiecare faČ›etÄ sunt Č›inute
Ă®n afara trainingului Č™i folosite pentru NLL/perplexity Č™i generare. Sunt
raportate validitatea strictÄ a contractului JSON, acoperirea cĂ˘mpurilor Ă®n
JSON Č™i Ă®n text, F1 lexical faČ›Ä de rÄspunsul de referinČ›Ä Č™i, pentru copil,
validitatea sintacticÄ Python. Scorul structural nu trebuie confundat cu
respectarea strictÄ a contractului JSON.

Rulare Ă®n mediul CUDA folosit de proiect:

```powershell
& 'C:\Users\xxsho\.conda\envs\licenta\python.exe' MI/benchmarks/model-family-smoke/run_benchmark.py
```

Se poate rula o singurÄ familie sau numai evaluarea:

```powershell
& 'C:\Users\xxsho\.conda\envs\licenta\python.exe' MI/benchmarks/model-family-smoke/run_benchmark.py --model qwen
& 'C:\Users\xxsho\.conda\envs\licenta\python.exe' MI/benchmarks/model-family-smoke/run_benchmark.py --model qwen --skip-training
```

Artefactele sunt scrise Ă®n `facets/mi-benchmark/<familie>/`, iar statisticile
Ă®n `MI/benchmarks/model-family-smoke/results/`.

Gemma foloseČ™te `2e-6`, Ă®n timp ce Qwen Č™i Llama folosesc `5e-5`. Un trial
Gemma cu rata comunÄ a trecut verificarea de identitate, dar a produs gradienČ›i
foarte mari Č™i degradare catastroficÄ held-out; rezumatul respins este pÄstrat
Ă®n `results/gemma-unstable-trial-summary.json`. Prin urmare, rezultatele dintre
familii sunt teste de fezabilitate configurate, nu un clasament controlat.
