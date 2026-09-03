# Spyre: build & deploy in OpenShift/Kubernetes

How to build the `scr` server image for IBM's Spyre AIU accelerator and run
it on an OpenShift cluster. See `CLAUDE.md` for the local (non-container)
build; this doc is about the containerized path.

## Build

`tools/docker/Dockerfile.spyre` is a multi-stage build (`cargo-chef` for
dependency caching) on top of the IBM-internal Spyre base image
(`icr.io/ibmaiu_internal/2.0/x86_64/spyre:<tag>`). It bakes SuperDSC support
for the models selected by the `SCRATCHY_MODEL_FEATURES` build arg (default:
`model/granite-3.1-2b-instruct,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel`
— the granite-3.1 2B and 8B instruct fp8 configs) into the binary — see `CLAUDE.md`'s
Build section for what each `arch-<name>`/`model/<stem>`/`quant/<preset>`
feature scopes.

Build it as an OpenShift binary build with:

```bash
scripts/build-spyre-image.sh [tag]   # tag defaults to "latest"
```

This script:
1. Creates the `scratchy-spyre` `BuildConfig` if it doesn't already exist
   (reused on later runs for layer caching), outputting to
   `image-registry.openshift-image-registry.svc:5000/<your-project>/scratchy-spyre:<tag>`.
2. Stages a minimal build context in a temp dir (`Dockerfile.spyre`,
   `Cargo.toml`/`Cargo.lock`/`build.rs`, `crates/`, `third_party/` with test
   dirs stripped) so the upload is ~50MB instead of the whole repo.
3. Runs `oc start-build --from-dir=<tmpdir> --follow`.

Requires `oc login` and an OpenShift project with binary-build permissions
first. `scripts/monitor-build.sh [build-name]` re-attaches to a build's logs
later if you didn't use `--follow`, or want to check on one from another
shell.

Check the pushed image's size:
```bash
oc get imagestreamtag scratchy-spyre:latest -o jsonpath='{.image.dockerImageMetadata.Size}{"\n"}'
```

## Deploy

The Spyre accelerator is requested as a device resource, not a GPU — the key
pod-spec pieces (reverse-engineered from other dev pods on the
`a1-vllm-spyre` project; there's no other source for these):

```yaml
spec:
  schedulerName: spyre-scheduler
  nodeSelector:
    spyre.node-classification: all
  containers:
  - name: app
    image: image-registry.openshift-image-registry.svc:5000/<project>/scratchy-spyre:latest
    resources:
      limits:
        ibm.com/spyre_pf: "1"
      requests:
        ibm.com/spyre_pf: "1"
```

The image's default `CMD` is `scr serve $MODEL`, with `MODEL` defaulting to
`RedHatAI/granite-3.1-2b-instruct-FP8-dynamic` (override via an `env:` entry
to serve a different model — it must be one of the configs baked in via
`SCRATCHY_MODEL_FEATURES` at build time).

**Non-root gotcha:** the runtime image doesn't set a `USER`, so under
OpenShift's default restricted SCC it runs as an arbitrary non-root UID with
no writable `$HOME`. Model downloads (`config.json`, weights) will fail with
`Permission denied (os error 13)` unless you set a writable cache dir:

```yaml
    env:
    - name: HOME
      value: /tmp        # or a mounted PVC, e.g. /work, for a persistent cache
    - name: HF_HOME
      value: /tmp/hf-cache
```

For anything beyond a one-off smoke test, point `HF_HOME` at a PVC instead of
`/tmp` so weights survive pod restarts — see the `shared-models` PVC (mounted
at `/models` on most `*-spyre-dev` pods in this project) or a personal
`*-work` PVC for the convention other dev pods use.
