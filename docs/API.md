# API și CLI MI

## Worker

```text
GET    /v1/health
GET    /v1/capabilities
POST   /v1/sessions
GET    /v1/sessions/{id}
DELETE /v1/sessions/{id}
PUT    /v1/sessions/{id}/weights/{version}
POST   /v1/sessions/{id}/rounds/{round}/compute
GET    /v1/sessions/{id}/rounds/{round}
GET    /v1/sessions/{id}/rounds/{round}/gradient
POST   /v1/sessions/{id}/rounds/{round}/ack
POST   /v1/sessions/{id}/actions/cancel
```

`GET` pe ponderi și gradienți acceptă un singur Range. Uploadul folosește
`Content-Range` și `X-Content-SHA256`. Identificatorii nu pot conține separatori
de cale.

## CLI

```text
j-mi worker serve
j-mi worker probe
```

Comenzile istorice `train` și endpointurile `/training/jobs*` sunt shims de
migrare și răspund `trainer_required`. Generate și suprafața compatibilă Ollama
rămân neschimbate.
