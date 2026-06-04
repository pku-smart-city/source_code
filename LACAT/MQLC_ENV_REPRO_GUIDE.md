# MQLC Environment Reproduction Guide (Offline-First)

This document explains how to reproduce the original `mqlc` runtime environment from scratch.
It specifically addresses the issues we encountered in practice:

- unstable network causing `pip` timeouts;
- `gym==0.20.0` failing with modern build toolchains;
- `torch==2.4.1+cu124`, `torch_scatter`, and `torch_sparse` not available from normal PyPI mirrors;
- installation and verification of local custom wheels (`highway_env_mqlc`, `rl_agents`).

---

## 1) Goal and assumptions

Goal: in a fresh environment (example name: `mqlc_new`), complete all checks below:

1. `import torch, torch_geometric, highway_env, rl_agents` works.
2. `gym.make("highway-v0")` and `env.reset()` work.
3. Core versions match the original environment (especially Python, Gym, and Torch CUDA stack).

Assumptions:

- OS: Linux (Ubuntu)
- Python: `3.10.x`
- Package managers: `conda + pip`
- Offline dependency folder: `./wheelhouse`
- Local project wheel folder: `./dist`

---

## 2) Files the author must provide

Before readers start, prepare and distribute:

1. `requirements.lock.txt` (frozen dependency list)
2. `wheelhouse/` (offline wheels/sdists)
3. `dist/highway_env_mqlc-*.whl`
4. `dist/rl_agents-*.whl`
5. (optional) `environment.yml` and platform notes

Recommended structure:

```text
MQLC/
  requirements.lock.txt
  wheelhouse/
    *.whl
    gym-0.20.0.tar.gz
    ...
  dist/
    highway_env_mqlc-1.0.0+mqlc-py3-none-any.whl
    rl_agents-1.0.dev0-py3-none-any.whl
```

---

## 3) How to publish `wheelhouse` and `whl` files via GitHub Releases

Because `wheelhouse` is large (for example 3.5 GB), do **not** commit it to git history.
Publish assets in GitHub Releases instead.

### Published release for this project (download here)

The offline bundles for this workflow are published on the fork **MangoSea/source_code**:

- **Release page:** [MQLC offline assets v1](https://github.com/MangoSea/source_code/releases/tag/mqlc-env-assets-v1)
- **Tag:** `mqlc-env-assets-v1`

**Assets included:** `wheelhouse.tar.gz.part.00`, `wheelhouse.tar.gz.part.01`, `dist_whl.tar.gz`, `RELEASE_SHA256SUMS.txt` (plus GitHub-generated source archives, which you do not need for pip offline install).

**Direct download base URL** (for scripts; replace `<filename>` with an asset name):

`https://github.com/MangoSea/source_code/releases/download/mqlc-env-assets-v1/<filename>`

Example (download all user assets into current directory):

```bash
TAG=mqlc-env-assets-v1
BASE=https://github.com/MangoSea/source_code/releases/download/${TAG}
for f in wheelhouse.tar.gz.part.00 wheelhouse.tar.gz.part.01 dist_whl.tar.gz RELEASE_SHA256SUMS.txt; do
  wget -c "${BASE}/${f}"
done
```

If the fork URL or tag changes in the future, update the link above and the `BASE` URL in your scripts.

---

### 3.1 Package and split large files

Run in project root:

```bash
tar -czf wheelhouse.tar.gz wheelhouse
split -b 1900M -d wheelhouse.tar.gz wheelhouse.tar.gz.part.
tar -czf dist_whl.tar.gz dist
sha256sum wheelhouse.tar.gz.part.* dist_whl.tar.gz > RELEASE_SHA256SUMS.txt
```

This produces:

- `wheelhouse.tar.gz.part.00`, `wheelhouse.tar.gz.part.01`, ...
- `dist_whl.tar.gz`
- `RELEASE_SHA256SUMS.txt`

### 3.2 Upload to GitHub Release

In your fork/repository:

1. Go to **Releases** -> **Create a new release**
2. Create a tag, e.g. `mqlc-env-assets-v1`
3. Upload all files listed above
4. Publish release

### 3.3 What readers download

Readers should download all release assets into the same directory:

- every `wheelhouse.tar.gz.part.*`
- `dist_whl.tar.gz`
- `RELEASE_SHA256SUMS.txt`

### 3.4 Verify and reconstruct archives

```bash
sha256sum -c RELEASE_SHA256SUMS.txt
cat wheelhouse.tar.gz.part.* > wheelhouse.tar.gz
tar -xzf wheelhouse.tar.gz
tar -xzf dist_whl.tar.gz
```

After extraction, readers should have local `wheelhouse/` and `dist/`.

---

## 4) Why installation may fail on a fresh machine

### 4.1 `gym==0.20.0` metadata/build errors

This is usually not your project code.
`gym 0.20.0` is old, and modern `pip/setuptools/wheel` + PEP517 build isolation can fail during metadata parsing/build steps.

### 4.2 `torch==2.4.1+cu124` not found on standard mirrors

CUDA-tagged wheels (`+cu124`) are not from regular PyPI.
They must come from the PyTorch index.

### 4.3 `pip download` shows progress but `wheelhouse` is empty

`pip` may download to temp locations first.
If the process crashes/timeouts before completion/checksum, files may never be finalized into `-d wheelhouse`.

---

## 5) Standard reproduction steps (reader side)

> Commands below assume current directory is project root: `/path/to/MQLC`

### Step 1) Create a new environment

```bash
conda create -n mqlc_new python=3.10 -y
conda activate mqlc_new
```

### Step 2) Pin packaging toolchain

```bash
python -m pip install -U "pip==23.1.2" "setuptools==59.6.0" "wheel==0.37.1"
```

Do not upgrade these tools immediately, otherwise old packages may fail again.

### Step 3) Install offline main dependencies

```bash
python -m pip install --no-index --find-links ./wheelhouse -r requirements.lock.txt
```

If `gym==0.20.0` was removed from `requirements.lock.txt`, continue with Step 4.
If gym is already installed successfully from Step 3, skip Step 4.

### Step 4) Install Gym separately (recommended)

```bash
python -m pip install ./wheelhouse/gym-0.20.0.tar.gz --no-build-isolation
```

### Step 5) Install local custom wheels

```bash
python -m pip install --no-deps ./dist/highway_env_mqlc-*.whl
python -m pip install --no-deps ./dist/rl_agents-*.whl
```

### Step 6) If missing Torch CUDA / PyG components, install them

If you see `No module named 'torch'` or `No module named 'torch_geometric'`:

```bash
python -m pip install "torch==2.4.1+cu124" --index-url https://download.pytorch.org/whl/cu124
python -m pip install "torch-geometric==2.6.1"
python -m pip install "torch-scatter==2.1.2+pt24cu124" "torch-sparse==0.6.18+pt24cu124" -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

`torch-geometric` extension wheels must match Torch and CUDA versions.

### Step 7) Final verification

```bash
python -c "import torch, torch_geometric, highway_env, rl_agents; print('all imports OK')"
python -c "import gym, highway_env; env=gym.make('highway-v0'); env.reset(); print('highway OK')"
```

Expected success output:

- `all imports OK`
- `highway OK`

---

## 6) Key takeaways (for README/paper)

1. Most failures in fresh environments come from package ecosystem/toolchain compatibility, not project code.
2. `gym==0.20.0` is more stable with `--no-build-isolation` and pinned older packaging tools.
3. `torch==2.4.1+cu124` cannot rely on normal PyPI mirrors only.
4. Most robust workflow: `wheelhouse + dist + fixed versions`.
5. Custom wheels `highway_env_mqlc` and `rl_agents` were validated in a new environment (import and `highway-v0` reset both succeed).

---

## 7) Author checklist before publishing

- [ ] `requirements.lock.txt` matches the final working environment
- [ ] `wheelhouse` includes `gym-0.20.0.tar.gz` (or equivalent installable source)
- [ ] `dist/` includes both local wheels
- [ ] CUDA version is clearly documented (`cu124`)
- [ ] Release assets include checksum file
- [ ] Verification commands and expected outputs are included

---

## 8) FAQ

### Q1: `ModuleNotFoundError: No module named 'highway_env'`
Check whether `highway-env-mqlc` was installed:
`python -m pip show highway-env-mqlc`

### Q2: `ModuleNotFoundError: No module named 'torch'`
Install `torch==2.4.1+cu124` from the PyTorch cu124 index.

### Q3: `ModuleNotFoundError: No module named 'torch_geometric'`
Install `torch-geometric` and matching `torch-scatter`/`torch-sparse` wheels.

### Q4: TensorFlow oneDNN / TF-TRT warnings appear
These are informational warnings in this context and do not block `highway_env` validation.

