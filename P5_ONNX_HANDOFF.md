# Handoff: add QAOA p=5 to the ONNX model zoo and publish

> **Status:** PAUSED pending real p5 `feature_normalization` stats. Export path is fully validated;
> publish deferred. Written 2026-09-03, progress updated 2026-09-04.

## PROGRESS (2026-09-04)

Done and verified (working tree, **uncommitted**):
- Export tooling recovered from `bd9d5a6^` (`tools/inference/export_onnx.py`, `gen_baselines.py`,
  `qaoa_training_pipeline/inference/torch_backend/`). Runtime remains torch-free — these are staged
  but not wired into the runtime import path.
- 7 p5 configs written at `inference/model_configs/<model>/p5/model_config.json`, cloned from p4 with
  `output_dim: 10` and the open-source `p_5` checkpoint path. Verified: p5 state-dict shapes match p4
  for every arch (only the head output differs 8→10), and the p4 checkpoint itself has `ff=128` while
  its config says `256` — so that param is ignored by the constructor; cloning p4 is safe.
- All 7 exported to `model.onnx`/`model.onnx.data` and **passed the parity gate** (max abs diff
  ≤1.2e-7 ≪ 1e-5). The torch-free runtime (`AIInference`) loads each p5 bundle and returns 10 angles.
- The `edge_transformer ` trailing-space checkpoint dir is referenced by its literal path in the config
  (checkpoints repo left untouched).

**BLOCKER — p5 `feature_normalization` is a PLACEHOLDER (p4 stats).** The real stats are not available
locally: not embedded in the p5 checkpoints, and only reachable via unmounted cluster paths
(`/dccstor/eevdata/cmahlasi/2026_AI_QAOA_graphs/results_p5/best_param_alltopology_p5.json` and the
`instance_dir`). Because normalization is applied in numpy *before* the ONNX graph, export and parity
pass regardless — wrong stats silently corrupt runtime predictions. Each p5 config carries a
`feature_normalization_status` field marking this; **do not commit or publish until it is replaced.**

**To resume once real p5 stats are in hand:**
1. Put the real p5 per-feature mean/std into all 7 configs' `feature_normalization`, then delete the
   `feature_normalization_status` marker from each.
2. Re-export is NOT required (graphs are norm-independent) — the existing `model.onnx` are final.
3. Freeze baselines: `.venv-export/bin/python tools/inference/gen_baselines.py`.
4. Publish (user-confirmed target = same HF repo as p1–4): `upload_to_hf.py` then `hf_manifest.py`.
5. Commit the 7 p5 `model_config.json` + updated `hf_manifest.json` + new baselines; keep the big
   `.onnx`/`.onnx.data` out of git (they live on HF).

---


## 0. The initial prompt (verbatim)

The request originally landed in the sibling `quantum_ai_parameter_prediction` repo:

> "I've added the P5 checkpoints
> `AI/quantum_ai_parameter_prediction_checkpoints_open_source`
> `github.ibm.com/IBM-Research-AI/quantum_ai_parameter_prediction_checkpoints_open_source`
> please also convert to onnx and publish"

**Course-correction:** the ONNX export + publish machinery does **not** live in
`quantum_ai_parameter_prediction`. It lives here, in `qaoa_training_pipeline`, and is more mature. So
this task belongs in **this** repo. The p=1..4 checkpoints→ONNX pipeline already exists here (and was
then archived); adding **p=5** means re-using that same pipeline for one more depth.

## 1. Goal

Add QAOA depth **p=5** (7 architectures) to the torch-free ONNX model zoo and publish it, matching
exactly how p=1..4 were produced. The p=5 **checkpoints** already exist at:

```
~/gitco/quantum_ai_parameter_prediction_checkpoints_open_source/p_5/<arch>/<name>.ckpt
```

(repo `git@github.ibm.com:IBM-Research-AI/quantum_ai_parameter_prediction_checkpoints_open_source.git`).

End state: 7 new bundles `qaoa_training_pipeline/inference/model_configs/<model>/p5/`, their
`.onnx`/`.onnx.data` uploaded to the HuggingFace Hub and pinned in `hf_manifest.json`, plus frozen
regression baselines under `test/inference/baselines/`. Zoo grows 28 → **35 bundles** (7 archs × p1..5).

## 2. How p=1..4 was done (this is the pattern to copy)

### Runtime (torch-free ONNX) — already in place, do not change
`qaoa_training_pipeline/inference/` runs exported ONNX graphs with `onnxruntime` + numpy — **no torch,
no checkpoints at runtime**. Three layers (see [`inference/README.md`](qaoa_training_pipeline/inference/README.md)):

- [`ai_inference.py`](qaoa_training_pipeline/inference/ai_inference.py) — `AIInference`, a
  `ProblemParamsProvider`; `provide_params(cost_op) → ParamResult` of β/γ angles.
- [`onnx_predictor.py`](qaoa_training_pipeline/inference/onnx_predictor.py) — `OnnxQAOAPredictor`;
  loads `model.onnx`, feeds only inputs the graph declares, validates `output_dim`.
- [`feature_extractor.py`](qaoa_training_pipeline/inference/feature_extractor.py) — numpy
  `extract_np`/`pack_features_np` turning a `SparsePauliOp` into model inputs (scalar features,
  edges/edge_weights, `rescale_a`).
- [`onnx_inputs.py`](qaoa_training_pipeline/inference/onnx_inputs.py) — `numpy_input_builders`
  registry: model type → the exact numpy feed dict its ONNX graph expects.
- [`model_registry.py`](qaoa_training_pipeline/inference/model_registry.py) — `ensure_onnx_local`,
  **local-first with lazy HuggingFace download** of the big weight files.
- [`config_io.py`](qaoa_training_pipeline/inference/config_io.py) — `load_config`,
  `resolve_bundle_path`.

Scale-invariance (must match training, or ONNX numbers drift): operator is normalized on input by
`rescaling_factor(cost_op)` (RMS of per-Pauli-order coeffs); on output predictions are `× π/2` then the
**gammas only** are divided back by `rescale_a`. Gated by `denormalize_output` (default `True`).

### Bundle layout
```
qaoa_training_pipeline/inference/model_configs/<model>/p<p>/model_config.json   # tiny, in git
                                              .../hf_manifest.json               # pins HF weights
```
Big `model.onnx` + `model.onnx.data` live on the **HF Hub**, not git. `<model>` ∈
`{mlp, edge_transformer, diffusion_transformer, gcn, graph_neural_network, graph_isomorphism_network,
graph_transformer}`. A "bundle key" is `<model>/p<p>` (e.g. `gcn/p3`). Note **`mlp/`** dir but its
config `model_type` is **`agg_transformer`**.

Sample p4 config (target schema — clone this for p5):
```jsonc
{
  "description": "Aggregation-style MLP for p=4 QAOA (trained across topologies, seed 42)",
  "checkpoint": "../../../../../../quantum_ai_parameter_prediction_checkpoints/multi_topology_checkpoints_new/p_4/mlp/mlp_p4_alltopo_seed42/mlp-p4-...ckpt",
  "model_init": { "model_type": "agg_transformer", "input_dim": 5, "output_dim": 8,
                  "embed_dim": 64, "num_layers": 4, "in_features": [ ... 5 graph stats ... ] },
  "feature_normalization": { "num_nodes": {"mean":62.18,"std":32.22}, ... },  // per-depth!
  "denormalize_output": true, "output_scale": 1.5707963267948966
}
```

### The one-time export path — REMOVED, must be recovered
The checkpoint→ONNX exporter and its torch backend were deleted in commit **`bd9d5a6`**
("Make inference ONNX-only: remove all torch code and backends") to keep the *runtime* torch-free.
Everything is recoverable from its parent `bd9d5a6^`. Files removed:

- `tools/inference/export_onnx.py` — the exporter (per-arch wrappers, torch dynamo export,
  `--check-parity` @ `atol=1e-5`, default `--opset 18`).
- `tools/inference/gen_baselines.py` — freezes `test/inference/baselines/`.
- `tools/inference/bench_onnx_vs_torch.py`, `tools/inference/compare_quality.py` — optional QA.
- `qaoa_training_pipeline/inference/torch_backend/` — the whole torch package: `model_loader.py`,
  `lightweight_predictor.py`, `inference_model.py`, `models.py`, and `ml_models/` with all 7
  architectures (`mlp`, `edge_transformer`, `ddpm_transformer`, `graph_convolutional_network`,
  `graph_neural_network`, `graph_isomorphism_network`, `graph_transformer`).

Inspect before use:
```
git show bd9d5a6^:tools/inference/export_onnx.py
git show bd9d5a6 --stat        # full list of what came out
```

### How each architecture was made ONNX-exportable (all 7 already work for p1..4)
`export_onnx.py` wraps each core model in a **fixed-signature `nn.Module`** whose inputs match
`numpy_input_builders[model_type]`, then exports at **B=1** with dynamic node/edge axes. The
ONNX-hostile ops are moved out of the graph into numpy precompute:

| Arch | Trick used in the export wrapper |
|---|---|
| `agg_transformer` (mlp) | none — pure `forward(x)`, dynamic batch. |
| `edge_transformer` | B=1 batch loop traces once; inputs `x, edges, edge_weights`, dynamic `num_edges`. |
| `diffusion_transformer` | eval **`t=0`** (no-noise) baked in, so `t` is not an ONNX input. |
| `graph_neural_network`, `graph_isomorphism_network` | `GraphListWrapper` rebuilds the single-element `[(edge_index, edge_attr)]` list from plain tensors at B=1. |
| `gcn` | skip PyG `Batch`/`to_data_list`; **per-node z-scored degree precomputed in numpy** (`node_x`); `global_mean_pool` = plain mean at B=1. |
| `graph_transformer` | **Laplacian PE (`torch.linalg.eigh`) and degree node-features precomputed in numpy** (`pos_enc`, `node_features`) and passed in; `edge_mask.fill_diagonal_` replaced by an `arange` index-put (fill_diagonal_ specializes num_nodes to a constant and breaks tracing). |

So there is **no eig / no PyG op inside any ONNX graph** — the exporter runs only the trained
projection/attention/head stack over a single graph.

### Environment
`~/gitco/qaoa_training_pipeline/.venv-export/` already has **torch 2.13 + torch_geometric 2.8 + onnx 1.22
+ onnxruntime 1.28** — the export env. The runtime-only extra is `pip install -e ".[inference]"`
(onnxruntime + huggingface_hub, no torch). Publish tooling: `tools/inference/hf_manifest.py`,
`tools/inference/upload_to_hf.py`, `tools/inference/model_keys.py`, `tools/inference/bench_ops.py`
(see [`tools/inference/README.md`](tools/inference/README.md)).

## 3. P5 checkpoint facts (verified by inspecting the .ckpt files)

- 7 architectures, **one seed each**. All share `input_dim=5`, `output_dim=10` (=2·p), `embed_dim=64`.
- Each `.ckpt` is a Lightning checkpoint; `hyper_parameters` carry `model_type`, `in_features`
  (the 5 graph stats: `num_nodes, num_edges, edges_per_node, mean_degree, std_degree`), `input_dim`,
  `output_dim`. Remaining arch dims (`num_layers`, `n_heads`, `edge_embed_dim`, `pos_enc_dim`, …) match
  the p4 configs and/or are inferable from state-dict shapes.
- `model_type` in the mlp checkpoint is `"mlp"` → use config `model_type` **`agg_transformer`**.
- Files (note the p5 date stamp `20260721`):

| dir | file |
|---|---|
| `mlp/` | `mlp-p5-20260721_144303-81-0.0005.ckpt` |
| `edge_transformer ` ⚠️ trailing space | `edge_transformer-p5-20260721_145654-87-0.0004.ckpt` |
| `diffusion_transformer/` | `diffusion_transformer-p5-20260721_144936-91-0.0004.ckpt` |
| `gcn/` | `gcn-p5-20260721_145654-52-0.0006.ckpt` |
| `graph_neural_network/` | `graph_neural_network-p5-20260721_150701-92-0.0002.ckpt` |
| `graph_isomorphism_network/` | `graph_isomorphism_network-p5-20260721_150017-90-0.0001.ckpt` |
| `graph_transformer/` | `graph_transformer-p5-20260721_151011-49-0.0006.ckpt` |

- ⚠️ **`p_5/edge_transformer ` has a trailing space in the directory name** (upload glitch). Either
  `git mv` it in the checkpoints repo, or reference the literal path (with the space) in the config.

## 4. Step-by-step to add p=5

1. **Recover the export path** onto a working branch (runtime stays ONNX-only):
   ```bash
   git checkout bd9d5a6^ -- tools/inference/export_onnx.py tools/inference/gen_baselines.py \
       qaoa_training_pipeline/inference/torch_backend
   ```
2. **Write 7 configs** `qaoa_training_pipeline/inference/model_configs/<model>/p5/model_config.json`,
   cloning the p4 schema, with:
   - `output_dim: 10`;
   - `checkpoint`: relative path to the **open_source** checkpoints repo
     `…/quantum_ai_parameter_prediction_checkpoints_open_source/p_5/<arch>/<file>.ckpt`.
     ⚠️ p4 configs point at the **private** `…_checkpoints/…` repo — recompute the `../` depth for p5
     and point at the open_source repo instead (or wherever you keep p5 locally).
   - `feature_normalization`: **fresh p5 stats — do NOT copy p4's.** p1 and p4 stats already differ
     (e.g. `num_nodes` mean 60.52 vs 62.18), so these are depth/split-specific. Source them from the p5
     training artifacts (the datamodule's computed norm stats / any `manifest_norm_stats`), and verify
     before trusting. Wrong stats silently corrupt predictions.
3. **Export** in `.venv-export`, one per arch, parity gate must pass:
   ```bash
   for m in mlp edge_transformer diffusion_transformer gcn graph_neural_network \
            graph_isomorphism_network graph_transformer; do
     .venv-export/bin/python tools/inference/export_onnx.py --model $m/p5 --check-parity
   done
   ```
   (writes `model.onnx` + `model.onnx.data` next to each p5 config; refuses to write on parity > 1e-5.)
4. **Freeze baselines** for p5: `.venv-export/bin/python tools/inference/gen_baselines.py`
   → new entries under `test/inference/baselines/`.
5. **Publish**:
   - `python tools/inference/upload_to_hf.py` to push the 7 new `.onnx`/`.onnx.data` to the HF Hub;
   - `python tools/inference/hf_manifest.py` to refresh `inference/model_configs/hf_manifest.json`
     (sha256 + size per file);
   - commit: the 7 p5 `model_config.json`, updated `hf_manifest.json`, new baselines. Keep the big
     weight files out of git (they live on HF).
   - ⚠️ **HF upload is outward-facing / hard to reverse — confirm the exact HF repo target and that
     publishing these open-source checkpoints is intended before running `upload_to_hf.py`.**
6. **Sanity check** end-to-end: build a `SparsePauliOp`, run `AIInference` on a `…/p5` bundle, confirm
   it returns **10** angles and matches the frozen baseline.

## 5. File map (quick reference)

| Area | Paths |
|---|---|
| Runtime (torch-free) | `qaoa_training_pipeline/inference/{ai_inference,onnx_predictor,onnx_inputs,feature_extractor,model_registry,config_io}.py`, `inference/README.md` |
| Bundles + manifest | `qaoa_training_pipeline/inference/model_configs/<model>/p<p>/model_config.json`, `.../hf_manifest.json` |
| Export tooling (recover from `bd9d5a6^`) | `tools/inference/{export_onnx,gen_baselines,bench_onnx_vs_torch,compare_quality}.py`, `qaoa_training_pipeline/inference/torch_backend/**` |
| Publish tooling | `tools/inference/{hf_manifest,upload_to_hf,model_keys,bench_ops}.py`, `tools/inference/README.md` |
| Regression baselines | `test/inference/baselines/` |
| p5 checkpoints (source) | `~/gitco/quantum_ai_parameter_prediction_checkpoints_open_source/p_5/<arch>/*.ckpt` |

## 6. Gotchas

- **p5 `edge_transformer ` dir has a trailing space** — normalize it.
- **Never copy p4 `feature_normalization` into p5** — stats are depth/split-specific; source the real
  p5 stats.
- **p5 `checkpoint` paths must point at the open_source checkpoints repo**, not the private one the p4
  configs reference.
- **Keep the runtime torch-free** — the recovered `torch_backend/` + `export_onnx.py` are one-time
  tooling; don't wire them into the runtime import path or add torch to the `inference` extra.
- **Parity gate is the guardrail** — if `--check-parity` fails for an arch at p5, the exported graph is
  wrong (usually a dynamic-axis or precompute mismatch); fix before publishing rather than lowering the
  tolerance.
- **Big weights belong on HF, not git** — only `model_config.json` + `hf_manifest.json` are committed.
