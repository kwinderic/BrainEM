# BrainEM

Reference implementation of **BrainEM**: eight neuron segmentation methods
evaluated on the paper's Same-Source and Cross-Source tracks. All eight share one
pipeline:

```
train -> infer -> post-process (waterz & agglomeration) -> eval (VOI/ARAND, ERL) -> visualize
```

Each method is one script under `exps/BrainEM/scripts/`; every stage is common
code in `exps/_lib/common.sh`. Run everything from the repo root.

## Tracks

The two tracks differ only in what a model is trained on and where it is then
evaluated; the architecture, objective and post-processing are identical.

| track | training set | validation | test volumes | script |
|---|---|---|---|---|
| Same-Source | `CremiAll` (CREMI A+B+C train splits) | CREMI A/B/C test splits | FlyWire | `<method>@cremiAll.sh` |
| Cross-Source | `TrainPoolAll` (multi-source pool) | pool val split | SNEMI, FIB25, Wafer4, M93 | `<method>@poolAll.sh` |

Same-Source trains on CREMI and tests on FlyWire, which comes from the same FAFB
source, so it measures accuracy under matched imaging conditions. The CREMI test
splits serve as the validation set during training (they select
`checkpoint_best.pth.tar`), and each method script also reports metrics on them
for reference. Cross-Source trains once on the pooled multi-source data and tests
on four datasets held out of that pool, measuring generalisation across imaging
conditions, resolutions and tissue with no per-target finetuning.

Every method has one script per track -- 16 in total, plus the two SSL
pretraining scripts.

## 1. Environment

```bash
conda create -n brainem python=3.10 -y && conda activate brainem
conda install pytorch=2.5 pytorch-cuda=12.4 torchvision -c pytorch -c nvidia -y
conda install numpy scipy h5py imageio tifffile opencv scikit-image scikit-learn \
              einops tqdm matplotlib yacs pyyaml tensorboard mahotas -c conda-forge -y
pip install fire tabulate wandb

pip install -e . --no-deps                                # for `import connectomics`
pip install git+https://github.com/zudi-lin/waterz.git    # agglomeration, not on PyPI
```

`--no-deps` installs `connectomics` as an editable package without re-resolving
dependencies, since the conda lines above already provide them.

FGNet additionally needs the SAM2 backbone weights:

```bash
bash projects/FGNet/sam2aff/sam2/checkpoints/download_ckpts.sh
```

Check:

```bash
python -c "import torch, connectomics, waterz, mahotas; print('cuda:', torch.cuda.is_available())"
```

ERL evaluation is optional and needs
[em_erl](https://github.com/PytorchConnectomics/em_erl) with `kimimaro`:

```bash
git clone https://github.com/PytorchConnectomics/em_erl && pip install -e em_erl
```

## 2. Data

Download the datasets from HuggingFace into `datasets/EM/`:

```bash
huggingface-cli download f74ksdfh/BrainEM --repo-type dataset --local-dir datasets/EM
```

Same-Source track -- CREMI for training and validation, FlyWire as the test target:

```
datasets/EM/cremi/sample_{A,B,C}_train_inputs.h5    # uint8,  (100, 1250, 1250)
datasets/EM/cremi/sample_{A,B,C}_train_labels.h5    # uint32, instance IDs
datasets/EM/cremi/sample_{A,B,C}_test_inputs.h5     # uint8,  (25, 1250, 1250)
datasets/EM/cremi/sample_{A,B,C}_test_labels.h5
datasets/EM/Flywire/flywire_crop_inputs.h5          # 400x4000x4000
datasets/EM/Flywire/flywire_crop_labels.h5
datasets/EM/Flywire/flywire_crop_small_inputs.h5    # 200x2000x2000 corner, quick checks
datasets/EM/Flywire/flywire_crop_small_labels.h5
```

Cross-Source track -- the pool file lists plus the four held-out volumes:

```
datasets/EM/Pool/train_pool_all_{images,labels}.txt  # one volume path per line
datasets/EM/Pool/val_pool_all_{images,labels}.txt
datasets/EM/AC3-AC4/AC3_{inputs,labels}.h5          # SNEMI  (100, 1024, 1024)
datasets/EM/FIB25/0{,_label}.tif                    # FIB25  (520, 520, 520), isotropic
datasets/EM/wafer4/wafer4_{inputs,labels}_test.h5   # Wafer4 (25, 1250, 1250)
datasets/EM/pyw93/9_3_78_97_{Stack,Labels}.tif      # M93    (20, 3960, 3990)
```

Images are `uint8`, labels are integer instance IDs with 0 as background, both in
`(z, y, x)` order, one dataset per file. FlyWire labels are the public consensus
segmentation rather than manual GT, so those numbers are best read as relative
comparisons between methods.

Paths live in `configs/datasets/<Name>.yaml`, which both bash and Python read, so
the two cannot disagree:

| yaml | role |
|---|---|
| `CremiAll` | Same-Source training pool (A+B+C train splits joined with `@`) |
| `CremiA` / `CremiB` / `CremiC` | CREMI test splits: validation during training, also scored |
| `Flywire` / `FlywireSmall` | Same-Source test targets (same FAFB source as CREMI) |
| `TrainPoolAll` | Cross-Source training pool (multi-source, via `.txt` volume lists) |
| `SNEMI` / `FIB25` / `Wafer4` / `M93` | Cross-Source test targets |
| `SSLPoolAll` | unlabeled pool for SSL pretraining |

To use your own data, edit the yaml, not the scripts.

## 3. Pipeline

One command trains and evaluates:

```bash
bash exps/BrainEM/scripts/superhuman@cremiAll.sh                     # Same-Source
bash exps/BrainEM/scripts/superhuman@poolAll.sh                      # Cross-Source
TRAIN_GPUS="1,2" bash exps/BrainEM/scripts/superhuman@cremiAll.sh    # pick GPUs
```

The two-stage methods pretrain first:

```bash
bash exps/BrainEM/scripts/dbmim_pretrain.sh      # -> outputs/dbmim_pretrain/
bash exps/BrainEM/scripts/dbmim@cremiAll.sh      # loads that encoder
```

### Stages

`exps/_lib/common.sh` exposes each stage separately, so any one can be re-run
without redoing the rest. All take a dataset name (a yaml stem in
`configs/datasets/`):

| function | does |
|---|---|
| `run_train [N]` | DDP training on `TRAIN_BASE_CONFIG` |
| `run_infer <Vol>` | sliding-window inference -> `result.h5` |
| `run_postprocess <Vol> [t]` | waterz -> `segments.npy`, then eval |
| `run_eval_only <Vol>` | metrics on an existing `segments.npy` |
| `run_vis <Vol>` | per-slice PNG figures |
| `run_sweep <Vol> [ts]` | one watershed, all thresholds -> csv |
| `eval_on <Vol> [t]` | infer + postprocess + vis (the default path) |
| `eval_segneuron <Vol>` | same, for SegNeuron's dual-head fused inference |

Env hooks: `TRAIN_GPUS` / `INFER_GPUS` to pin cards, `CHECKPOINT` to point
inference at a specific file. With `TRAIN_GPUS` unset, `wait_gpus` waits for idle
cards. Batch size is per rank, so global batch = GPUs x `SAMPLES_PER_BATCH`.

**Train** writes `checkpoint_<iter>.pth.tar` every `SOLVER.ITERATION_SAVE`, plus
`checkpoint_best.pth.tar` when a validation loader exists. dbMiM and SegNeuron
finetuning build none, so their eval points at the final numbered checkpoint.
Stdout and the script itself are teed into `<WORK_DIR>/log.txt`.

**Infer** is a sliding window with Gaussian blending; patch and stride come from
`MODEL.INPUT_SIZE` / `INFERENCE.STRIDE`. Peak GPU memory is independent of volume
size (0.45 GB for SuperHuman, 2.0 GB for FGNet), so the same command handles the
0.8 Gvox crop and the 6.4 Gvox volume. Output is `result.h5`: `uint8`, key `vol0`,
shape `(3, z, y, x)`.

**Post-process** runs a seeded watershed (`maxima_distance`) for fragments, then
waterz mean-affinity agglomeration, giving `segments.npy`. This is a whole-volume
CPU pass and the memory bottleneck: roughly 28 bytes per voxel, so about 180 GB of
RAM for the full FlyWire crop. CREMI volumes need under 1 GB. To choose a
threshold use `run_sweep` -- one watershed, all thresholds walked incrementally.

**Eval** reports VOI split/merge/sum and adapted RAND (lower is better):

```bash
python scripts/eval.py eval_full <gt_labels.h5> <segments.npy> [result.h5]
```

VOI split penalises over-segmentation and VOI merge under-segmentation; a method
trades one for the other via the waterz threshold, which is why the sweep matters
when comparing methods. Passing `result.h5` adds voxel-level affinity accuracy.

ERL (expected run length, higher is better) weights errors by tracing cost, so a
merge between two long neurites counts far more than a small boundary slip. Build
the GT skeleton graph once per volume, then score each segmentation:

```bash
cd /path/to/em_erl
python examples/seg_to_graph.py -s <gt_labels.h5> -r 1,1,1 -t 8 -o <graph>.npz
python examples/eval_volume.py -p <segments.npy> -g <graph>.npz -r 1,1,1
```

`-r 1,1,1` reports ERL in voxels, following em_erl's j0126 workflow. Within a
dataset this rescales every method by the same constant, so rankings hold; ERL is
not comparable across datasets. For volumes where skeletonization exhausts memory,
`scripts/build_erl_graph_batched.py` batches over objects instead of space (set
`EM_ERL=/path/to/em_erl` if it is not pip-installed).

**Visualize** writes one PNG per sampled slice: raw image, GT, predicted
affinity, predicted segmentation.

```bash
python scripts/vis_slices.py --image <inputs.h5> --label <labels.h5> \
    --affinity <WORK_DIR>/<Vol>/result.h5 --segment <WORK_DIR>/<Vol>/segments.npy \
    --output_dir <WORK_DIR>/<Vol>/vis
```

Outputs per test volume, under `exps/BrainEM/outputs/<method>/<subset>/<Volume>/`:
`config.yaml` (fully resolved), `result.h5`, `segments.npy`, plus
`segments_t<t>.npy` and `sweep_waterz.csv` when sweeping, and `vis/`.

### Reproduced result (One-run)

SuperHuman trained from scratch with this repo on the Same-Source track, scored on
the CREMI test splits (the validation set) at waterz threshold 0.5:

| volume | voi_split | voi_merge | voi_sum | ARAND |
|---|---:|---:|---:|---:|
| CREMI A | 0.500 | 0.202 | 0.701 | 0.100 |
| CREMI B | 0.835 | 0.158 | 0.993 | 0.043 |
| CREMI C | 1.125 | 0.187 | 1.311 | 0.157 |

## 4. Model configurations

Config comes from three layers, later overriding earlier: the dataset yaml
(`--config-base`), the method yaml (`--config-file`), and trailing `KEY VALUE`
pairs from each script's `MODEL_EXTRA_ARGS`. Every key with its default is in
`connectomics/config/defaults.py`. The table lists the effective value after
overrides.

| Method | ARCH | config | entry point | patch (z,y,x) | stride | optimizer | LR | iters | GPUs x batch/rank | stages |
|---|---|---|---|---|---|---|---:|---:|:---:|---|
| SuperHuman | `superhuman` | [SUPERHUMAN-BASE.yaml](projects/Baselines/configs/base/SUPERHUMAN-BASE.yaml) | `Baselines/main.py` | 18x160x160 | 10x80x80 | Adam | 1e-4 | 200k | 2 x 1 | supervised |
| PEA | `pea` | [PEA-BASE.yaml](projects/Baselines/configs/base/PEA-BASE.yaml) | `Baselines/main.py` | 18x160x160 | 10x80x80 | Adam | 1e-4 | 100k | 2 x 1 | supervised |
| MALA | `mala` | [MALA-BASE.yaml](projects/Baselines/configs/base/MALA-BASE.yaml) | `Baselines/main.py` | 84x268x268 -> 56x56x56 | 56x56x56 | Adam | 1e-4 | 100k | 2 x 1 | supervised |
| CAD | `cad` | [CAD-BASE.yaml](projects/Baselines/configs/base/CAD-BASE.yaml) | `Baselines/main.py` | 18x160x160 | 10x80x80 | Adam | 1e-4 | 200k | 2 x 1 | supervised |
| AGQ | `affinity_knet` | [AGQ-Base.yaml](projects/AGQ/configs/AGQ-Base.yaml) | `AGQ/main.py` | 17x257x257 | 9x128x128 | Adam | 1e-4 | 100k | 8 x 1 | supervised |
| dbMiM | `dbmim_unetr_aniso_em` | [dbMiM-Base.yaml](projects/dbMiM/configs/dbMiM-Base.yaml) | `dbMiM/main.py` | 32x160x160 | 16x80x80 | AdamW | 8e-5 (enc 1e-5) | 160k + 20k | 4 x 2 | SSL -> finetune |
| SegNeuron | `segneuron` | [SegNeuron-Base.yaml](projects/SegNeuron/configs/SegNeuron-Base.yaml) | `SegNeuron/main.py` | 20x128x128 | 10x64x64 | Adam | 1e-3 | 100k + 100k | 4 x 2 | SSL -> finetune |
| FGNet | `sam2aff_pt_fg` | [FGNet-16x256.yaml](projects/FGNet/configs/FGNet-16x256.yaml) | `FGNet/main.py` | 16x256x256 | 8x128x128 | Adam | 1e-4 | 100k | 4 x 1 | supervised |

### Targets and losses

Five methods optimise the same single 3-channel affinity target with the same
loss, so only the architecture differs. The other three keep the objective that
defines them:

| Method | target(s) | loss |
|---|---|---|
| SuperHuman, FGNet | `["2"]` -- 3-ch affinity | WeightedBCEWithLogits + 0.5 Dice |
| AGQ | `["9-1","2"]` -- instance label + affinity | BCE + 0.5 Dice (affinity head) |
| dbMiM | `["2"]` | WeightedMSE |
| SegNeuron | `["2","0"]` -- affinity + foreground | WeightedMSE, both heads |
| MALA | `["2"]` | MalisLoss |
| PEA, CAD | 4 multi-scale affinity targets | computed internally (PEA: SCM+CCM+EPM; CAD: 3D/2D/cross/interact) |

For PEA and CAD the `LOSS_OPTION` entries exist only so the framework's
`Criterion` builds; both return their own loss from `forward()`. Shared across all
eight: `MODEL.LABEL_EROSION 1`, waterz at the same threshold, one eval script.

`MODEL.FILTERS` must match the checkpoint being loaded: `update_checkpoint` warns
on shape-mismatched layers and skips them, so a mismatch infers with randomly
initialised weights for those layers.

### Adding a method

Copy `exps/_lib/template_method.sh` into `exps/BrainEM/scripts/` as
`<method>@<subset>.sh` and set `ARCH`, `MAIN_PY`, `CONFIG`, `MODEL_EXTRA_ARGS`,
`TRAIN_BASE_CONFIG` and the `eval_on` lines. The filename determines the output
path. Register a new architecture with the project's registration decorator,
which inserts the class into `MODEL_MAP` under the name used as `ARCH`.

## Layout

```
connectomics/          framework: models, data, trainer, config
projects/              one directory per method: main.py + configs/ + model code
configs/datasets/      dataset paths, read by both bash and python
scripts/               post_process.py, eval.py, vis_slices.py, sweep_waterz.py
exps/_lib/             pipeline: common.sh, env.sh, template_method.sh
exps/BrainEM/scripts/  the 16 method scripts and the two SSL pretrainings
exps/BrainEM/outputs/  checkpoints and predictions (gitignored)
datasets/              volumes (gitignored; see section 2)
```

## Citation

```bibtex
@article{brainem,
  title   = {},
  author  = {},
  journal = {},
  year    = {}
}
```

## Acknowledgement

Built on [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics),
with agglomeration by [waterz](https://github.com/zudi-lin/waterz) and ERL by
[em_erl](https://github.com/PytorchConnectomics/em_erl). The `projects/`
directories reimplement or adapt SuperHuman, PEA, MALA, CAD, AGQ, dbMiM,
SegNeuron and SAM2 within this framework.
