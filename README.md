# Jester Mixed Inference — componenta 2/5

MI definește și execută fațete structurale peste același model de bază. Folosește
blocuri complete de decoder inspirate de block expansion/LLaMA Pro, sloturi
explicite sau uniforme și compoziție reversibilă `base + părinți + copil`.

Componentele Jester sunt LSR, MI, RM, Web Interface și Trainer.

## Responsabilități

- sloturi și mai multe blocuri per slot;
- DAG-uri cu unul sau mai mulți părinți;
- inițializare identity-copy/copy/reset/zero;
- artifacte compacte safetensors cu numai blocurile fațetei;
- generate și suprafață HTTP/Ollama-compatible;
- `mi.worker`: model load, sesiuni versionate și gradient numai în copil;
- Range/resume, SHA-256, zstd și dtype configurabil;
- binding managed CPU/CUDA și refuz fail-closed pentru backend necompatibil.

MI nu mai deține joburi, optimizer sau scheduler. Endpointurile istorice
`/training/jobs` răspund `trainer_required`; migrarea se face cu:

```powershell
j-train migrate-mi-config old-training.json campaign.json `
  --topology-id training-pool --dataset-id coder-data
j-train migrate-mi-checkpoint old-output migrated-initial.safetensors
```

Workerul este pornit numai de RM prin `j-mi worker serve`. Generate și
compoziția MI rămân independente de Trainer. Baza și părinții sunt înghețați în
worker, iar Trainer aplică optimizerul și agregarea sincronă.

Documentația se află în [docs](docs), iar contractele în [schemas](schemas).

## Limite declarate

- un worker folosește maximum un accelerator logic;
- NPU fără autograd are numai fallback CPU explicit;
- XLA/TPU nu este `hardwareVerified` fără test real;
- compoziția generate este serializată în interiorul unei instanțe pentru
  consistența mutării reversibile a blocurilor.

## Shared SDK runtime

Managed-channel validation is shared through `jester_sdk_runtime`; the existing
component CLI/API and resource checks remain in place. Native builds include
this support as compiled code. Recipients of the standard Engine Native Blocks
do not install the authoring SDK separately.

Source development and build hosts require the matching SDK runtime sources
(`JesterWorkspace/jester-sdk/src` on the development import path), plus Core
and its RM client. See the [SDK documentation](../../../../jester-sdk/README.md)
and [workspace build recipes](../../../../j-compile/recipes/README.md). This is a build
dependency, not a new runtime package installation step.
