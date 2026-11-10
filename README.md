# Bird species identification

A small PyTorch training and batch prediction project for fine-grained bird classification. It uses an ImageNet initialized ResNet-50 and works with either the official CUB-200-2011 files or a conventional class-folder dataset.

## Project layout

```text
bird-species-identification/
├── dataset.py          # CUB and class-folder datasets, transforms, loaders
├── model.py            # ResNet-50 and backbone freezing controls
├── train.py            # training, validation, scheduler, checkpoints
├── predict_batch.py    # batched prediction to CSV
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Set up the environment

Use Python 3.10 or newer. Create a virtual environment and install the packages:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for your machine if you want GPU training. The CPU build also works, though training will take substantially longer. The first pretrained run downloads ResNet-50 ImageNet weights.

## 2. Prepare the data

### CUB-200-2011

Extract the dataset so the directory passed to `--data-dir` contains the annotation files and `images/` directory:

```text
data/CUB_200_2011/
├── images/
├── classes.txt
├── images.txt
├── image_class_labels.txt
└── train_test_split.txt
```

The official train/test assignment in `train_test_split.txt` is used for training and validation. The test split is treated as validation here; for a final reported result, keep a separate validation split or report test performance only after model selection is finished.

### Class-folder dataset

Arrange images in this structure (class names become labels):

```text
data/birds/
├── train/
│   ├── cardinal/
│   └── blue_jay/
└── val/
    ├── cardinal/
    └── blue_jay/
```

The same class names must appear in both splits. Add any number of class folders; the model head size is inferred from the training split.

## 3. Train

For CUB-200-2011:

```bash
python train.py --data-dir data/CUB_200_2011 --dataset cub --epochs 30 --batch-size 32
```

For a class-folder dataset:

```bash
python train.py --data-dir data/birds --dataset folder --epochs 30 --batch-size 32
```

The training transforms resize/crop to 224 pixels, apply horizontal flips and mild color jitter, then normalize with ImageNet statistics. Validation uses a deterministic 224×224 resize and the same normalization. The loop uses cross-entropy with light label smoothing, AdamW, cosine annealing, and mixed precision on CUDA. It saves `checkpoints/last.pth` each epoch, `checkpoints/best.pth` when validation accuracy improves, and the class list in `checkpoints/classes.json`.

To train only the classifier first, then allow the later ResNet stages to update:

```bash
python train.py --data-dir data/CUB_200_2011 --freeze-backbone --unfreeze-from layer4
```

`--unfreeze-from` accepts `layer4`, `layer3`, `layer2`, `layer1`, or `all`. When freezing is disabled (the default), all layers train. On smaller GPUs, lower `--batch-size`; `--workers 0` can help on constrained Windows setups.

## 4. Predict on an image directory

```bash
python predict_batch.py --input-dir data/unseen --checkpoint checkpoints/best.pth --output predictions.csv --batch-size 128
```

Inference batches images and uses CUDA when available. The CSV contains one row per image for top-1, or one row per class candidate with `--top-k 5`. Confidence is the softmax probability for that candidate. Input images can be nested below the directory.

## Practical training notes

The requested ~89% validation accuracy is a target, not a guarantee. Results depend on the split, data quality, hardware, training duration, and tuning. Begin with 20–30 epochs and inspect the validation trend. Fine-grained CUB classification often benefits from staged fine-tuning: train the new head briefly with the backbone frozen, then unfreeze `layer4` or the whole network with a lower learning rate. Keep the evaluation split untouched while tuning if you need an unbiased final score.

This starter reports top-1 accuracy and average loss. For a paper or benchmark, also document the exact split, seed, software/hardware, checkpoint selection rule, and any class imbalance handling.
