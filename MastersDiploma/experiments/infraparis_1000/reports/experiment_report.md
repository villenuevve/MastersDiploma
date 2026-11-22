# SegFormer-B3 Fine-tuning Experiment

## 1. Objective

Evaluate domain adaptation of the pretrained SegFormer-B3 model
to the InfraParis dataset for semantic segmentation.

## 2. Dataset

- Dataset: InfraParis
- Original Train candidates: 6567
- Usable Train images: 6566
- Images used in this experiment: 1000
- Excluded fully unlabeled Train images: 1
- Validation images: 189
- Independent Test images: 571
- Number of semantic classes: 19
- Unlabeled target pixels mapped to ignore index: 255

## 3. Model and Training Configuration

- Base model: `nvidia/segformer-b3-finetuned-cityscapes-1024-1024`
- Optimizer: AdamW
- Learning rate: 1e-05
- Weight decay: 0.01
- Batch size: 1
- Epochs: 3
- Maximum input side: 512 px
- Random seed: 42
- Device: mps
- Training time: 18.99 minutes
- Checkpoint selection: Best validation mIoU

## 4. Validation Results

| Stage | Training Loss | mIoU (%) | Pixel Accuracy (%) |
|---|---:|---:|---:|
| Baseline | — | 45.76 | 76.57 |
| Epoch 1 | 0.37827 | 49.53 | 78.49 |
| Epoch 2 | 0.27330 | 50.64 | 78.88 |
| Epoch 3 | 0.23699 | 51.58 | 81.11 |

## 5. Independent Test Results

| Metric | Original model (%) | Fine-tuned model (%) | Change (pp) |
|---|---:|---:|---:|
| mIoU | 52.55 | 58.85 | +6.30 |
| Pixel Accuracy | 84.68 | 88.42 | +3.74 |

The fine-tuned model improved IoU in 17 of 19 classes.

## 6. Interpretation

The fine-tuned model achieved higher mIoU and Pixel Accuracy than
the original pretrained model on the same independent Test split.

The Test split was not used for gradient updates or checkpoint selection.
The best checkpoint was selected using Validation mIoU.

The results indicate improved adaptation to the InfraParis domain.
The experiment used 1,000 of the 6,566 usable Train images, so it does
not represent training on the complete available training split.

## 7. Reproducibility

- Training configuration: `training_config.json`
- Test metrics: `results/segformer_b3_infraparis_1000_512_e3_test_metrics.json`
- Checkpoint: `models/segformer_b3_infraparis_1000_512_e3_filtered`
- Training image list: `tables/training_images.csv`

## 8. Generated Materials

See the `tables/` and `graphs/` directories for CSV tables and PNG figures.
