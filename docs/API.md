# CLI și API MI

## CLI

Comenzile principale sunt:

```text
jester-rm instance start <MI_TRAIN_INSTANCE>
jester-rm instance start <MI_GENERATE_OR_SERVE_INSTANCE>
python mi.py generate PROMPT --server URL --facet FACET
python mi.py plan --registry DIR --facet FACET
python mi.py facets --registry DIR
python mi.py migrate-rm-config INPUT OUTPUT
```

Comenzile locale `train`, `generate`, `serve` și `inspect-base` cer un lease RM și răspund `RM_LEASE_REQUIRED` dacă sunt pornite direct. Fațetele/planurile care nu încarcă un model și apelurile către un server existent rămân disponibile direct.

Administrarea serverului folosește `health`, `base-load`, `base-unload`, `facet-load`, `facet-unload`, `registry-refresh`, `train-submit`, `jobs` și `job [--cancel]`. Toate răspunsurile administrative sunt JSON.

## HTTP

Serverul ascultă implicit numai pe `127.0.0.1:8120`. Nu include autentificare sau TLS; nu trebuie expus direct în rețea. Contractul formal este în `schemas/openapi.yaml`.

| Metodă și rută | Rol |
|---|---|
| `GET /health` | bază, registry, memorie și joburi |
| `GET /facets` | artifacte disponibile/încărcate |
| `POST /base/load` | încarcă sau înlocuiește baza |
| `POST /base/unload` | eliberează baza și fațetele |
| `POST /registry/refresh` | rescanează manifestele |
| `POST /compose` | rezolvă DAG-ul fără inferență |
| `POST /facets/load` | încarcă anticipat fațeta și strămoșii |
| `POST /facets/unload` | elimină artifacte din cache |
| `POST /generate` | compune și generează |
| `POST /training/jobs` | legacy unmanaged; într-un `mi.serve` managed răspunde `409 rm_training_instance_required` |
| `GET /training/jobs` | listează joburile |
| `GET /training/jobs/{id}` | stare, evenimente și rezultat |
| `POST /training/jobs/{id}/cancel` | cere anularea |

### Generate

```json
{
  "schemaVersion": "mi-generate-v1",
  "requestId": "agent-turn-42",
  "messages": [{"role": "user", "content": "Scrie o funcție Python."}],
  "facets": ["coder.python@0.1.0"],
  "generation": {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "do_sample": true
  }
}
```

În loc de `facets`, payloadul LSR poate conține `resolvedActiveFacets`, `directSelectedFacets`, `selectedFacets` sau un obiect `route` care conține unul dintre acestea.

### Training legacy și tranziția RM

Pentru server pe aceeași mașină este preferată calea absolută, deoarece păstrează rezolvarea căilor relative din config:

```json
{"configPath": "E:\\Licenta\\Proiect\\MI\\examples\\train-coder-uniform.json"}
```

Se poate trimite și `{"config": {...}}` sau direct documentul `mi-training-v1`; în aceste cazuri căile interne trebuie să fie absolute pentru un server remote.

Stările jobului legacy sunt `queued`, `waiting-for-resource`, `running`, `cancelling`, `cancelled`, `succeeded` și `failed`.

Într-un proces pornit ca `mi.serve` de Resource Manager, endpointul de training refuză imediat cererea; nu creează un job care eșuează ulterior și nu primește tokenul privilegiat al controllerului. Operatorul creează o instanță separată `mi.train`, cu propriul lease, volum și buget. Endpointul rămâne temporar numai pentru compatibilitatea library/unmanaged și va putea fi eliminat după migrarea consumatorilor.

### Erori

Erorile așteptate au forma:

```json
{"error": {"code": "incompatible_artifact", "message": "..."}}
```

Codurile HTTP diferențiază configurația invalidă (`400`), resursa lipsă (`404`), incompatibilitatea/starea (`409`), body prea mare (`413`) și eroarea internă (`500`).
