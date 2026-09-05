# Model development

This guide covers the local setup for dataset preparation, training, export,
and evaluation. The complete model design and training contract live in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Quick start

Run these commands from the repository root:

```bash
uv sync
uv run python -m scripts.aggregator
uv run python -m model_shared.smoke --seconds-budget 60
```

The aggregator is idempotent. Use `--force` to rebuild the staged dataset or
`--validate` to validate an existing manifest:

```bash
uv run python -m scripts.aggregator --force
uv run python -m scripts.aggregator --validate
```

Open [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb) to run the full
training and evaluation workflow. Checkpoints are written under the ignored
`checkpoints/` directory and analysis JSON and plots are written under
`model_analysis/`.

## Model components

- `model_localizer/` predicts a bounding box from positive few-shot episodes.
- `model_siamese/` predicts whether the requested object is present.
- `model_shared/` contains dataset, fold, checkpoint, runtime, logging, and
  analytics utilities used by both models.
- `scripts/` contains dataset aggregation, inference, and ONNX/TFLite export
  entry points.

## Datasets

- [VizWiz Fewshot](https://vizwiz.org/tasks-and-datasets/object-localization)
  provides the localizer baseline.
- [HOTS](https://github.com/gtziafas/HOTS) provides household-object training
  and test data.
- [InsDet](https://insdet.github.io) provides instance-detection training and
  test data.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for stage-by-stage configuration,
augmentation rules, checkpoint durability, inference modes, and success
criteria.
