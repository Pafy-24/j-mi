# MI model-family smoke benchmark

Acest benchmark verifică end-to-end Componenta 2/4 pe trei arhitecturi mici:
Qwen2, Gemma 3 și Llama 3.2. Nu este un benchmark de calitate generală a
modelelor și nu poate valida specializare reală cu numai 14 exemple de
antrenare per fațetă.

Rezultatul executat și interpretarea critică sunt în `results/REPORT.md`, iar
valorile și outputurile brute sunt în `results/summary.json` și fișierele per
familie.

Pentru fiecare model sunt antrenate și salvate două fațete:

- `coder@smoke-2026-08-23`: două blocuri în două sloturi, aproximativ la 1/3
  și 2/3 din decoder;
- `coder.python@smoke-2026-08-23`: un bloc la aproximativ 5/6 din decoder,
  dependent de `coder`, care rămâne activ și înghețat la trainingul copilului.

Comparația are exact trei stări:

1. `base`;
2. `base+facet` (`coder`);
3. `base+parent+child` (se cere numai `coder.python`, iar DAG-ul rezolvă
   automat părintele).

Primele patru exemple SFT și toate cele zece exemple de continued pretraining
intră în training. Ultimele două exemple SFT din fiecare fațetă sunt ținute
în afara trainingului și folosite pentru NLL/perplexity și generare. Sunt
raportate validitatea strictă a contractului JSON, acoperirea câmpurilor în
JSON și în text, F1 lexical față de răspunsul de referință și, pentru copil,
validitatea sintactică Python. Scorul structural nu trebuie confundat cu
respectarea strictă a contractului JSON.

Rulare în mediul CUDA folosit de proiect:

```powershell
& 'C:\Users\xxsho\.conda\envs\licenta\python.exe' MI/benchmarks/model-family-smoke/run_benchmark.py
```

Se poate rula o singură familie sau numai evaluarea:

```powershell
& 'C:\Users\xxsho\.conda\envs\licenta\python.exe' MI/benchmarks/model-family-smoke/run_benchmark.py --model qwen
& 'C:\Users\xxsho\.conda\envs\licenta\python.exe' MI/benchmarks/model-family-smoke/run_benchmark.py --model qwen --skip-training
```

Artefactele sunt scrise în `facets/mi-benchmark/<familie>/`, iar statisticile
în `MI/benchmarks/model-family-smoke/results/`.

Gemma folosește `2e-6`, în timp ce Qwen și Llama folosesc `5e-5`. Un trial
Gemma cu rata comună a trecut verificarea de identitate, dar a produs gradienți
foarte mari și degradare catastrofică held-out; rezumatul respins este păstrat
în `results/gemma-unstable-trial-summary.json`. Prin urmare, rezultatele dintre
familii sunt teste de fezabilitate configurate, nu un clasament controlat.
