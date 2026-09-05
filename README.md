# Looking Glass

Looking Glass is a cross-platform object-finding app. A user teaches the app
an object with a small set of photos, then points a phone camera at a
real-world scene and asks Looking Glass to find it.

The project combines a Flutter mobile client with a Go inference backend and a
custom ONNX-based few-shot vision pipeline. The system can recognize whether
the requested object is present and, when it is, return its bounding box in
the camera image.

## What makes it few-shot?

Looking Glass does not require a large collection of photos for every object a
user might want to find. Instead, the user supplies 1-10 support images for a
category, such as "my keys" or "the red toolbox." Those images become the
temporary visual description of the object.

At search time, the AI compares the live query image with the support images.
The number of support images can vary from one to ten, so the system can use
more examples when an object has different viewpoints, lighting, or
appearances.

## How the app works

The user flow is intentionally simple:

1. Create a category and capture a few photos of the target object.
2. Choose that category in **Find Object**.
3. Capture a scene with the camera.
4. Let the backend decide whether the object exists in the scene and where it
   is located.

The Flutter app handles category management, camera capture, settings, and
results. The backend stores support images, loads the ONNX models, runs
inference, and returns the existence score and bounding box. See
[`APPLICATION.md`](APPLICATION.md) for the technology stack, setup, API, and
runtime data flow.

## Quick start

There are two complementary workflows:

- To run the product, follow [`APPLICATION.md`](APPLICATION.md). It covers
  model assets, ONNX Runtime, the Go backend, the Flutter app, local-network
  configuration, and Android builds.
- To prepare data, train, evaluate, or export models, follow
  [`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md). It explains the commands and
  points to [`MODELS_ARCHITECTURE.md`](MODELS_ARCHITECTURE.md) for the full
  model and checkpoint contract.

For a quick model-environment check from the repository root:

```bash
uv sync
uv run python -m scripts.aggregator
uv run python -m model_shared.smoke --seconds-budget 60
```

The smoke test expects the aggregated dataset described in
[`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md). The application cannot start
until the three exported ONNX assets described in
[`APPLICATION.md`](APPLICATION.md) are available.

## AI pipeline

The runtime uses a cascade of three vision stages:

```text
Support images + query image
          |
          v
RMBG-1.4 salient-object preprocessing
          |
          v
Multi-shot Siamese existence model
          +--> object probably absent -> return "not found"
          |
          v
Multi-shot localizer
          |
          v
Predicted bounding box
```

### 1. Salient-object preprocessing

RMBG-1.4 is used to emphasize the salient foreground object before comparison.
This helps reduce the influence of distracting backgrounds and makes the
support examples more useful to the downstream models.

The data pipeline preserves the full image geometry with letterbox resizing.
Support images are not cropped using their bounding boxes; this keeps
preprocessing consistent with the way users capture examples in the app.

### 2. Siamese existence check

The Siamese model answers: "Does the requested object appear in this query
image?" It produces an `existence_prob` value from 0 to 1.

Its backbone is a frozen DINOv2-small vision transformer. A trainable
cross-attention pool lets query-image patches attend to the patches from all
support images. The model also receives similarity statistics between query
and support patches, including:

- maximum similarity;
- top-five similarity mean and standard deviation;
- mean similarity;
- CLS-token similarity; and
- similarity entropy.

These features are combined by a small MLP head. The existence model is
deliberately separate from localization: it can reject likely negatives
before the more expensive localizer runs.

### 3. Multi-shot localizer

The localizer answers: "Where is the object?" It predicts a normalized
bounding box in `(cx, cy, w, h)` format.

Its frozen backbone is OWLv2 (`google/owlv2-base-patch16-ensemble`). Each
support image is converted into OWLv2's native image-query embedding. The
embeddings are then fused with a small trainable transformer:

- a learnable `[CLS]` token summarizes the support set;
- padded support slots are ignored with a key-padding mask;
- no positional embeddings are used, so the result is invariant to
  support-image order; and
- a residual identity path starts from the mean support embedding and learns a
  correction on top of it.

The residual correction is scaled by a trainable `alpha`, initialized at
`0.01`. This gives the model a stable starting point close to OWLv2's
mean-support behavior while allowing support fusion to improve it. If fusion
is not helpful, training can reduce the correction toward zero.

The localizer predicts patch scores first, selects the patch nearest the best
match, and uses the corresponding box prediction as the final result.

## Combined inference

The default combined mode is a hard cascade:

1. Run the Siamese model.
2. Compare `existence_prob` with the configured existence threshold.
3. Skip localization and return no box when the score is below the threshold.
4. Run the localizer only when the score passes the threshold.

Two additional modes are available for analysis and product behavior:

The modes are:

- `hard`: gate localization on the Siamese threshold.
- `soft`: always run both models and mark low-confidence boxes below the
  threshold.
- `always_localize`: always return the localizer box with the Siamese score.

The threshold can be swept on the test split to study the trade-off between
false positives, false negatives, and localization quality.

## What is custom?

The project uses strong pre-trained vision backbones, but the few-shot
behavior and the way the models cooperate are custom:

- support images are fused dynamically for each query rather than converted
  into a fixed class list;
- the Siamese model uses cross-attention over all support-image patch tokens;
- handcrafted patch-similarity statistics complement the learned features;
- the localizer uses a permutation-invariant support transformer and a
  residual prototype correction;
- existence and localization are trained and evaluated independently;
- combined inference exposes an adjustable cascade threshold; and
- the backend packages the models behind a single mobile-friendly REST API.

RMBG-1.4, DINOv2-small, and OWLv2 provide the visual foundations. The
trainable support-fusion layers, classification heads, localization heads,
LoRA adapters, losses, episode construction, checkpoint format, and combined
inference logic are project-specific.

## Fine-tuning strategy

The two learned models are trained independently. This makes it possible to
improve or evaluate recognition without changing localization, and vice
versa.

### Localizer stages

The localizer is trained only on positive episodes because the Siamese model
owns the existence decision.

The stages are:

- **Phase 0:** no fine-tuning; establish the zero-shot OWLv2 baseline.
- **L1:** train support fusion, `[CLS]`, and `alpha` with patch
  classification loss.
- **L2:** add class and box heads plus layer normalization for direct box
  regression with L1 and GIoU losses.
- **L3:** add LoRA adapters to the last four OWLv2 blocks while keeping most
  of OWLv2 frozen.

The localization objective is:

- patch cross-entropy for the patch nearest the ground-truth center;
- L1 box loss in `(cx, cy, w, h)` format; and
- GIoU loss in `(x1, y1, x2, y2)` format.

The patch objective is important because it forces the support prototype to
discriminate between image patches. It avoids a degenerate solution that
assigns similar scores everywhere.

### Siamese stages

The Siamese model is trained on positive and negative episodes, with a
default ratio of one positive to three negatives.

The stages are:

- **Phase 0:** no fine-tuning; establish the frozen DINOv2 baseline.
- **S1:** train the cross-attention pool, similarity features, and MLP head
  for the existence decision.
- **S2:** add LoRA adapters to the last four DINOv2 blocks for selected
  query/value projections.

The Siamese objective combines:

- focal BCE, with extra emphasis on avoiding false positives;
- a variance regularizer to prevent pooled representations from collapsing;
  and
- a decorrelation regularizer to keep representation dimensions from
  learning the same signal.

Misclassified negatives are placed in a hard-negative cache. Later epochs
sample part of their negatives from this cache, allowing training to focus on
difficult scenes instead of repeatedly seeing only easy examples.

## Dataset and episode construction

The training data is aggregated into a validated schema-version-6 manifest.
The current sources are:

- [HOTS](https://github.com/gtziafas/HOTS): training and testing for
  household objects in tabletop scenes.
- [InsDet](https://insdet.github.io): training and testing for instance
  detection.
- [VizWiz Fewshot](https://vizwiz.org/tasks-and-datasets/object-localization):
  Localizer Phase 0 only, as a baseline source.

HOTS and InsDet are split into train and test partitions using an 80/20
stratified split. Training episodes sample a variable number of support
images, `K in {1, ..., 10}`, uniformly. Evaluation uses a deterministic
round-robin over representative support counts, including `K=1`, `K=4`, and
`K=10`, so multi-shot behavior can be compared consistently.

Each episode contains:

```text
support images + support mask + actual K
query image
query bounding box (positive episodes)
presence label
instance ID and dataset source
native image size and native bounding box for diagnostics
```

Negative episodes are used by the Siamese model. Their query comes from a
different instance in the same source, which creates realistic "similar but
not the target" examples. The localizer sees positive episodes only.

### Image preprocessing and augmentation

All images are resized with aspect ratio preserved and padded to a square
using a mean color. During training, support images may receive horizontal
flips, random resized crops, color jitter, grayscale conversion, blur, and
random erasing. Query images receive only mild color jitter so their spatial
layout remains valid for bounding-box supervision. Evaluation and Phase 0 use
letterboxing without training augmentation.

Each model applies its own normalization internally because the DINOv2 and
OWLv2 pipelines use different image statistics.

## Evaluation

The models use metrics suited to their separate responsibilities:

- localizer: mAP@50, mAP@50:95, IoU statistics, containment diagnostics, and
  per-`K`/per-source breakdowns;
- Siamese: AUROC, PR-AUC, false-positive rate, false-negative rate, accuracy,
  and Brier score; and
- combined pipeline: threshold sweeps showing FPR/FNR/accuracy and mAP@50 on
  positives that pass the existence gate.

The project's target criteria include localizer mAP@50 of at least 20% on
InsDet and 30% on HOTS, Siamese AUROC of at least 0.80, Siamese FPR at
threshold 0.5 of at most 10%, and improved localizer mAP when moving from one
support image to ten.

## Checkpoints and experiment durability

Training checkpoints contain only trainable weights rather than a duplicate
copy of each frozen backbone, keeping them relatively small. A checkpoint
also records the optimizer, scheduler, mixed-precision scaler, resolved
configuration, fold plan, metrics history, early-stopping state, and
random-number-generator state.

Stages can resume from `last.pt`, restore the best validation checkpoint, or
warm-start automatically from the previous completed stage. Per-epoch
snapshots, `best.pt`, `last.pt`, and `stage_complete.pt` support both recovery
and inspection of experiments.

When training on Google Colab, output roots are required to be on the mounted
Google Drive. Writes use temporary files, flushing, filesystem
synchronization, atomic replacement, and retries so that a runtime
interruption is less likely to lose a checkpoint or analysis file.

## Repository layout

```text
.
├── app/                    Flutter mobile and desktop client
├── backend/                Go REST API and inference services
├── models/                 Local ONNX assets (ignored by Git)
├── scripts/                Dataset, inference, and model-export entry points
├── notebooks/              Training and evaluation notebooks
├── model_analysis/         Tracked evaluation results and plots
├── model_localizer/        Multi-shot localization model and training code
├── model_siamese/          Multi-shot existence model and training code
├── model_shared/           Dataset, checkpoint, logging, and analytics tools
├── APPLICATION.md          Application setup and runtime guide
├── MODELS_ARCHITECTURE.md  Model and runtime architecture
├── MODEL_DEVELOPMENT.md    Model-development quick start
├── pyproject.toml          Python dependency and tool configuration
└── uv.lock                 Locked Python dependencies
```

The `models/`, `dataset/`, `checkpoints/`, `exports/`, and `inference/`
directories are local runtime areas and are intentionally ignored. The
tracked `model_analysis/` directory contains reproducible experiment summaries
and plots.

The application/runtime details are in [`APPLICATION.md`](APPLICATION.md),
the model design is in
[`MODELS_ARCHITECTURE.md`](MODELS_ARCHITECTURE.md), and the development
workflow is in [`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md).
