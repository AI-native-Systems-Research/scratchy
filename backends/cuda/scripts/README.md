# The H100 drivers

Each run is one Kubernetes pod with one H100: `drive.sh` creates it from `pod-template.yml`, copies
a run script and its helpers in, waits for the script's last line, copies the logs and results
out, and deletes the pod, always (also on failure or Ctrl-C).

## What to set

| variable | for | default |
|---|---|---|
| `NAMESPACE` | the namespace the pod runs in | required |
| `KUBE_CONTEXT` | the kubectl context | kubectl's current one |
| `WORKDIR` | where `logs/<pod>/` is written | the current directory |
| `IMAGE` | the pod's image (CUDA 12.8+ devel) | `nvidia/cuda:12.9.1-devel-ubuntu22.04` |
| `NODE_SELECTOR_KEY`, `NODE_SELECTOR_VALUE` | the H100 node pool | `cloud.google.com/gke-accelerator`, `nvidia-h100-80gb` (GKE) |
| `TOLERATION_KEY`, `TOLERATION_VALUE` | the pool's `NoSchedule` taint | `nvidia.com/gpu`, `present` (GKE) |
| `HF_SECRET` | a Kubernetes Secret holding a Hugging Face token under the key `token` (optional: the pinned checkpoints are public) | `hf-token` |
| `REPO_URL` | the git repository the pod clones | required |
| `SHA` (`kernels.sh`), `SHA_A`, `SHA_B` (`e2e.sh`) | full commit SHAs | required |
| `BASE` (`kernels.sh`) | a second SHA whose GPU suite is compared with `SHA`'s | none |
| `FETCH_REFS` | branches to fetch first, when a SHA is not otherwise fetchable | none |
| `BASELINE_COHERENCE` | a local `coherence-scr.json` from an earlier run, whose tokens `e2e.sh` compares against | none |

A token is never written to a file: the pod reads it from the Secret, as `HF_TOKEN`.

## The order

```bash
NAMESPACE=… REPO_URL=… SHA=… BASE=… ./drive.sh scratchy-cuda-k kernels.sh -
NAMESPACE=… REPO_URL=… SHA_A=… SHA_B=… ./chain.sh
```

1. `kernels.sh`, the GPU suite at `SHA`: the build, the ThunderKittens source check, ptxas over
   every kernel (0 spills), every CUDA test binary built with `nvcc` (and at `BASE`, compared by
   `ab.py`), the FP8 tape tests, and the CLI build, which must link no cuBLAS or cuDNN.
2. `chain.sh`, the end-to-end A/B: `e2e.sh` once per model (`MODEL=27b|35b|gemma`, the checkpoint
   pinned to `pins/`), one pod each, one after another. Both SHAs build in one pod and serve on the
   same GPU, A1, B1, A2, B2; vLLM serves the same checkpoint in the same pod as the reference.
   `report.py` writes `results/ab.md`.

Either SHA may predate `backends/cuda`: both scripts build the CUDA crates from the root workspace
then.

`accuracy_vs_vllm.py` is the accuracy evidence against vLLM over HTTP (an agreement set and a
teacher-forced score of `testdata/`'s text), run against one server at a time; its docstring has
the usage.

## The log lines

Every step appends to the pod's `/logs/exits.txt`, copied to `logs/<pod>/exits.txt`:

- `EXIT <step> <code> WALL=<s>`: one step, its exit code and wall time (`/logs/<step>.log` holds
  its output);
- `STOP: <why>`: a step the run cannot go on without failed; the script ends there;
- `K_DONE` (`kernels.sh`), `MODEL_DONE` (`e2e.sh`): the script's last line, which `drive.sh` waits
  for;
- `GATE A vs B tokens identical: <n>/<m> prompts PASS|FAIL` (`e2e.sh`): whether A's and B's
  coherence tokens match;
- `POD GONE` (`drive.sh`): the pod disappeared mid-run; `NOT RUN: … already exists`: a pod of that
  name was already there, so nothing ran and nothing was deleted.
