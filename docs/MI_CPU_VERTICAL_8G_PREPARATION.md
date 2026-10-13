# MI CPU vertical preparation (8 GiB)

Status: offline preparation only. This document does not authorize a service,
lease, model load, inference, or hardware probe.

The fixed base is `Qwen/Qwen2.5-1.5B-Instruct`, revision `main`, from the
already imported model asset
`sha256:17eca20be0dc7099db78e03edfb75bd9f060c6a29f9c6dd22540bf31a4584d66`.
The loader-input manifest is
`sha256:501a7e29596c225926dd04f4a06923b7a014b377b3ab66a3696d305052ded08e`
and the verified model artifact identity is
`7f8b34632910df9fc87f5924ee695fb59770ff50a5857d073a2ae8e14a92bee0`.
The model safetensors are BF16, so the bounded CPU request must use explicit
`bfloat16`. The previous `float32` candidate is invalid under 8 GiB: MI's
pre-construction bound includes the 3,087,428,608 source tensor bytes and the
6,174,857,216 expanded target bytes.

The fixed facet registry is
`sha256:76a4697c781169b2057d1a7d495dd7497b31eaaba26e96d0e6a024892541bff4`.
LSR remains the only facet-selection authority. For the fixed prompt it must
return `coder.python@smoke-2026-08-23`; Engine's existing
`validate-mi-request` transform produces an MI request whose requested facet is
that child and whose expected resolver closure is, in order,
`coder@smoke-2026-08-23`, `coder.python@smoke-2026-08-23`.

## Resource policy

- Backend and device: RM-managed CPU, `requestedBackend=cpu`,
  `effectiveBackend=cpu`, no accelerator and no fallback.
- CPU: one exclusive logical processor, quota 1000 millicores.
- RAM: 8,589,934,592 bytes; swap: zero.
- MI operational margin: 1,073,741,824 bytes.
- Storage: 21,474,836,480 bytes.
- Service concurrency: one active request, zero queued requests. Two requests
  are sent sequentially through distinct Engine runs.
- Persistence: both runs target the same MI deployment generation, instance,
  process birth, active lease, and RM channel destination. There is no second
  materialization, install, process start, base load, facet-registry refresh,
  or runtime-tree verification between them.
- Generation: deterministic, seed 42, at most 16 new tokens per request.

The offline tensor-header accounting is:

| Item | Source bytes | BF16 target bytes |
| --- | ---: | ---: |
| Base | 3,087,428,608 | 3,087,428,608 |
| `coder` facet | 187,191,296 | 187,191,296 |
| `coder.python` facet | 93,595,648 | 93,595,648 |

The base-load peak plus the 1 GiB margin is 7,248,599,040 bytes, below the
8 GiB lease by 1,341,335,552 bytes. After the base is resident, staging both
facets plus the same margin is 4,722,744,320 bytes before the bounded request
buffers and KV cache. These are admission bounds from immutable safetensors
headers, not observed RSS and not live qualification evidence.

## Fixed requests and expected evidence

The Engine request template is fixed without `facets`:

```json
{"schemaVersion":"mi-generate-v1","prompt":"Write a Python function that returns the sum of two integers.","generation":{"max_new_tokens":16,"do_sample":false},"seed":42}
```

Engine must transform the structured `lsr-route-v2.selectedFacets` object into
the MI request. Request 1 must report two cache misses and commit both resolved
facets. Request 2 uses a different Engine run/idempotency identity with the same
prompt and LSR result against the still-running MI instance; it must reach MI
again and report two cache hits, zero
misses, zero loaded bytes, and the same composition. Replaying request 1 is
only an Engine idempotency check and does not count as the warm request.

If the first Engine→LSR→MI vertical succeeds, the minimal live extension is one
additional run payload before cleanup. It reuses the already resolved
deployment and channel. If the first request fails or becomes uncertain, the
second request is skipped and cleanup starts; it is never a retry. No separate
warm benchmark runner is introduced.

Each response must be structured `mi-generate-v1`, contain non-empty text and
1..16 output tokens, preserve the requested child and ordered two-facet
closure, and carry the real base/composition provenance. Engine must persist
both generation artifacts. The second output may differ only where the runtime
contract permits it; with the fixed seed and deterministic generation it is
expected to equal the first output.

## Existing artifact and remaining live prerequisite

Reuse the installed transitional bundle at
`.jester-runtime/autonomous-mi-cpu-cp313-win-amd64-20261013-06`; do not rebuild
it. Its independently reviewed assembly receipt records content digest
`a0347312a0483dd31aa1ad141d4a17d911be35d9fcc5feebe6b7ed15698b56f3`,
manifest SHA-256
`d487592b1e918d967cb8450e55fc8a8136ddf437fdf00e0c1a84bcb41e058bab`,
18,741 files, installed-only MI/Torch imports, and CLI help exit 0.

Before a live vertical, that exact bundle still needs a reviewed RM
`RuntimeArtifact` envelope and materialization identity. The existing
`mi-outer-envelope-plan-06.candidate.json` is preparation only; its output
reference must replace the older 2026-10-12 MI artifact in the lifecycle
manifest. A current CPU qualification and an authenticated approval bound to
that exact artifact, 8 GiB policy, host epoch, and evidence digest are also
required.

The live command remains the reviewed lifecycle runner with its effect gate:

```powershell
& '<pinned-rm-python>' -I -B `
  'DOCS/implementation-evidence/2026-10-12/mi-runtime-lifecycle-preparation-01/mi_cpu_lifecycle_v2.py' `
  --manifest '<root-reviewed-8g-manifest.json>' --execute-reviewed
```

This command may run only after explicit user approval for the first MI live
probe. The supervisor deadline must include startup, both sequential requests,
and cleanup. No automatic retry is allowed.

Cleanup must attempt Engine stop and MI stop independently after every start
attempt, await asynchronous stop operations, then prove terminal instances,
empty RM process inventories, exact leases durably released, owned process and
listener absence, zero active leases, and clean SQLite. Any unavailable check,
forced cleanup, identity drift, fallback, or deadline exhaustion is terminal
failure or `cleanup-uncertain`, never success.
