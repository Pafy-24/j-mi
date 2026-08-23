# Jester Mixed Inference (MI)

MI este componenta **2 din 4** a proiectului Jester. Ea antrenează, salvează, compune și execută fațete structurale peste același model de bază. Implementarea curentă folosește extensia prin blocuri de decoder inspirată de LLaMA Pro: blocurile noi sunt copiate din model, inițializate ca transformări identitate, iar la antrenare modelul de bază și fațetele-părinte rămân înghețate.

Cele patru componente planificate sunt:

1. LSR — rutarea către fațete;
2. MI — training și generate prin compoziție dinamică;
3. Agentic Platform & Resource Management;
4. Web Interface.

## Ce este implementat

- sloturi explicite sau uniforme, ancorate la indicii blocurilor modelului de bază;
- oricâte blocuri într-un slot și oricâte sloturi într-o fațetă;
- arbori/DAG-uri cu zero, unul sau mai mulți părinți (`requires`, `after`, `before`, `conflicts`, `priority`);
- antrenarea unei fațete copil în contextul `base + toți strămoșii`, cu gradient numai în copil;
- inițializare `identity-copy`, `copy`, `reset` sau `zero`, cu ținte configurabile;
- optimizer, scheduler, precizie, acumulare, clipping, checkpoint, resume și trainer Python custom opt-in;
- artifacte compacte `safetensors` care conțin doar blocurile fațetei;
- compoziție reversibilă pentru fiecare cerere de generare;
- CLI local/remote și API HTTP cu joburi asincrone de training și anulare;
- consum direct al selecțiilor LSR (`resolvedActiveFacets`, `directSelectedFacets`, `selectedFacets`).

## Pornire rapidă

```powershell
cd MI
python -m pip install -e .
python mi.py --PUs
python mi.py train examples/train-coder-uniform.json --json
python mi.py serve --registry artifacts/facets --base-model C:\models\base-1b --device cuda:0
python mi.py generate "Scrie o funcție Python" --facet coder.python --server http://127.0.0.1:8120
```

Exemplele au căi demonstrative pentru model și output; modelul de bază trebuie înlocuit cu unul local sau cu un ID Hugging Face și `allowDownload`/`--online` activat explicit.

Documentația detaliată este în:

- [Arhitectură](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [CLI și API](docs/API.md)
- [Surse și influențe](THIRD_PARTY_NOTICES.md)

## Limită intenționată a versiunii 1

MI v1 implementează fațete formate din blocuri complete de decoder pe un singur device. Nu face încă tensor/pipeline parallel, offload coordonat, batching între compoziții sau eviction după buget; acestea aparțin componentei 3. Compoziția aleasă este fixată înainte de prefill și rămâne neschimbată până la terminarea decodării, astfel încât KV cache-ul dinamic rămâne coerent.
