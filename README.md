# Jester Mixed Inference (MI)

MI este componenta **2 din 4** a proiectului Jester. Ea antrenează, salvează, compune și execută fațete structurale peste același model de bază. Implementarea curentă folosește extensia prin blocuri de decoder inspirată de LLaMA Pro: blocurile noi sunt copiate din model, inițializate ca transformări identitate, iar la antrenare modelul de bază și fațetele-părinte rămân înghețate.

Cele patru componente planificate sunt:

1. LSR — rutarea către fațete;
2. MI — training și generate prin compoziție dinamică;
3. RM — Resource Manager pentru lease-uri, containment și execuție locală/SSH;
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
- consum direct al selecțiilor LSR (`resolvedActiveFacets`, `directSelectedFacets`, `selectedFacets`);
- integrare obligatorie cu RM pentru `train`, generate local, `serve` și `inspect-base`;
- binding managed CPU/CUDA, buget cooperative VRAM și telemetrie per proces;
- procese separate `mi.train`, `mi.generate` și `mi.serve`; iGPU/NPU sunt refuzate fără fallback până la un backend MI real.

## Pornire rapidă

```powershell
cd E:\Licenta\Proiect\RM
python -m pip install -e ".[nvidia,windows-ml]"
jester-rm resources list --json
jester-rm instance create examples\instance-mi-generate.json
jester-rm instance plan mi-generate-01
jester-rm instance start mi-generate-01
jester-rm operations inspect <OPERATION_ID>
jester-rm instance logs mi-generate-01
```

Exemplul RM are căi și ID-uri hardware demonstrative. Modelul de bază, storage pool-ul și UUID-urile trebuie adaptate inventarului local. Lansarea directă a unei comenzi locale care încarcă modelul răspunde `RM_LEASE_REQUIRED`; `migrate-rm-config` convertește configurațiile vechi la `baseModel.device=managed`.

Documentația detaliată este în:

- [Arhitectură](DOCS/ARCHITECTURE.md)
- [Training](DOCS/TRAINING.md)
- [CLI și API](DOCS/API.md)
- [Surse și influențe](THIRD_PARTY_NOTICES.md)

## Limită intenționată a versiunii 1

MI v1 implementează fațete formate din blocuri complete de decoder pe un singur device logic. RM poate controla vizibilitatea mai multor plăci CUDA/XLA, dar MI standard nu face încă tensor/pipeline parallel, offload coordonat, batching între compoziții sau eviction de blocuri. Compoziția aleasă este fixată înainte de prefill și rămâne neschimbată până la terminarea decodării, astfel încât KV cache-ul dinamic rămâne coerent.
