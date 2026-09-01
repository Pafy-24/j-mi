# Facet-efficiency golden experiment

The experiment compares exactly the same base model and decoding parameters in
four states:

```text
base
base + coder
base + coder + coder.python
base + LSR route + MI composition
```

It uses 24 manually authored, multilingual architecture cases and all 164
HumanEval tasks. Neither suite is allowed in Trainer manifests or LSR
signatures. The runner never executes generated code; it only emits official
HumanEval sample files. Functional execution is a separate restricted Docker
step.

`miUrl` must point to an RM-managed `mi.serve` API (normally port 8120), not to
an `mi.worker` training endpoint (normally port 8130). For a remote MI service,
use the source-side loopback endpoint of an RM inference channel. Finish and
release the training pool before reserving the same accelerator for evaluation.

```powershell
python .\MI\benchmarks\facet-efficiency\run_golden.py run `
  --config .\MI\benchmarks\facet-efficiency\experiment.a40.json `
  --output .\MI\benchmarks\facet-efficiency\results\<run-id>

.\deploy\docker\evaluate-humaneval.ps1 `
  -RunDirectory .\MI\benchmarks\facet-efficiency\results\<run-id>

python .\MI\benchmarks\facet-efficiency\run_golden.py summarize `
  --run .\MI\benchmarks\facet-efficiency\results\<run-id>
```

`run.json` includes Wilson intervals, paired bootstrap deltas, MI timing and
control-plane snapshots from RM and Trainer. Every state also records exact
start/finish timestamps plus MI health before/after, including loaded facets and
CUDA allocator memory. A run with only generated-template NLL is not considered
evidence of facet efficiency.

Set `rm.metricsLimit` in the experiment file to bound each control-plane
snapshot. The default is 500 samples so provenance capture does not turn into a
multi-megabyte, slow request during a long golden run.
