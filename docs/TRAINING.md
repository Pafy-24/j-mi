# Training prin Jester Trainer

MI nu mai lansează joburi și nu aplică optimizerul. Configurația structurală a
fațetei rămâne `mi-training-v1`, dar este inclusă într-un `Campaign` Trainer.

```text
Trainer -> POST /v1/sessions -> mi.worker
Trainer -> PUT weights/version
Trainer -> POST round/compute
Trainer <- GET gradient (Range + SHA-256)
Trainer -> optimizer/scheduler/checkpoint
```

Workerul activează `base + părinți + copil`, îngheață baza și părinții și
calculează gradientul mediu al copilului ponderabil prin numărul real de tokeni
supravegheați. Un request stale sau reutilizarea unei runde cu alt hash este
refuzată.

Pentru stații remote, datasetul, modelul și registrele de părinți sunt
bindinguri CAS din manifestul RM. Căile locale nu sunt presupuse comune.

Migrare:

```powershell
j-train migrate-mi-config old.json campaign.json --topology-id pool --dataset-id data
j-train migrate-mi-checkpoint old-output initial.safetensors
```
