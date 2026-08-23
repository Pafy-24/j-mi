# Configurarea trainingului MI

Fișierul de intrare folosește `schemaVersion: mi-training-v1`. Exemple complete există în `examples/` și schema validabilă în `schemas/mi-training-v1.schema.json`.

## Controalele operatorului

| Zonă | Controale principale |
|---|---|
| Bază | ID/path, revision, `layerPath`, dtype, device, acces online, kwargs model/tokenizer |
| Structură | sloturi explicite sau uniforme, poziție, ordine, blocuri/slot, strat-sursă, initializer |
| Ierarhie | părinți multipli, constrângeri before/after, conflicte, prioritate |
| Date | mai multe fișiere/directoare, filtre de stage, repeat, limit, packing, prompt masking |
| Optimizare | optimizer, LR, weight decay, betas/eps/momentum, scheduler, warmup |
| Buclă | epochs/max steps, batch, gradient accumulation/checkpointing, precizie, clipping, seed |
| Siguranță | identity check și toleranțe, hash artifact/model, output overwrite explicit |
| Persistență | checkpoint periodic, optimizer state, resume |
| Extensie | trainer/optimizer/scheduler custom cu import explicit opt-in |

## Dataseturi acceptate

Loaderul citește JSON și JSONL, inclusiv direct un folder de fațetă. Când primește un folder, include fișierele `.json`/`.jsonl`, dar exclude `test.jsonl`.

Formatele recunoscute sunt:

- `facet-record-v3` cu stage `continued-pretraining`: causal language modeling pe `input.content`;
- `facet-record-v3` SFT: mesajele din `input` formează promptul, iar `target.messages` formează ținta;
- `{ "text": "..." }` pentru continued pretraining;
- `{ "prompt": "...", "completion": "..." }` pentru SFT;
- `{ "messages": [...] }` pentru conversații complete.

Cu `maskPrompt: true`, tokenii promptului primesc label `-100`. Pentru chat se folosește template-ul tokenizerului; fallback-ul textual există pentru tokenizere fără template.

Aceleași fișiere `facet-record-v3` pot fi proiectate în embeddings de LSR. Nu trebuie create exemple „pentru router” separate de exemplele fațetei.

## Ierarhie de training

Pentru un copil, părinții trebuie să existe deja în `registry` ca artifacte MI. De exemplu:

```json
"dependencies": {
  "requires": ["coder@0.1.0", "reasoner@0.1.0"]
}
```

Bucla obține compoziția `base + coder + reasoner + copil`, îngheață primele trei părți și optimizează numai copilul. Astfel se pot construi succesiv `base`, `base+parent`, `base+mai mulți părinți` și `base+parent+child+grandchild` fără a fuziona permanent ponderile.

## Inițializare

- `identity-copy`: copiază stratul-sursă și zeroizează proiecțiile reziduale de ieșire; implicit recomandat;
- `copy`: copie exactă, deci schimbă funcția și cere de regulă `identityCheck: false`;
- `reset`: apelează inițializatoarele modulelor leaf;
- `zero`: zeroizează toate ponderile blocului, util doar experimental.

Modelele necunoscute pot configura `identityTargets` cu nume exacte sau globuri. Trainingul se oprește dacă nu găsește nicio proiecție ori dacă verificarea logits eșuează.

## Checkpoint și resume

Un checkpoint conține:

```text
facet.mi.json
facet.safetensors
trainer-state.pt       # opțional
training-result.json   # numai outputul final
checkpoints/
  step-00000100/
```

`resumeFrom` indică folderul checkpointului sau direct `facet.mi.json`. Structura sloturilor, baza, ID-ul, versiunea și dependențele trebuie să coincidă cu configurația curentă. Starea optimizerului este reluată dacă `trainer-state.pt` există.

Verificarea identitară este executată numai la materializarea inițială. La resume este marcată `skipped: resumed-trained-weights`, deoarece după primul update o fațetă antrenată nu mai trebuie să fie identitatea.

## Algoritmi custom

Bucla implicită este causal LM. Pentru control complet asupra algoritmului se poate declara:

```json
{
  "allowCustomClasses": true,
  "algorithm": {
    "type": "experimental",
    "trainerClassPath": "my_package.MyTrainer",
    "kwargs": {"objective": "custom"}
  }
}
```

Clasa primește `TrainingConfig`, `progress_callback`, `cancel_event`, plus `kwargs`, și trebuie să expună `run() -> dict`. Optimizerele și schedulerele au extensii similare. API-ul refuză implicit importurile custom; pornește serverul cu `--allow-custom-training-classes` numai pe un host de încredere.
