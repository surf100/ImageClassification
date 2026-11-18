
# Computer Vision — Homework 2

## Image Classification with Transfer Learning

University of Verona  
Course: Computer Vision  
Professor: Francesco Setti  
Academic Year: 2026/2027

## 1. Project Objective

This project compares three image classification
strategies using the Oxford-IIIT Pet dataset
and a pretrained MobileNetV3-Small network.

The task is to classify 37 cat and dog breeds.

The three approaches are:

1. Full Fine-Tuning
2. Classifier-Head Training
3. Feature Extraction with Linear SVM

## 2. Dataset

Dataset: Oxford-IIIT Pet

Official source:
https://www.robots.ox.ac.uk/~vgg/data/pets/

The dataset contains 7,349 images from 37 breeds.

Data partitions:

| Split | Images |
|---|---:|
| Training | 2,944 |
| Validation | 736 |
| Test | 3,669 |

The official trainval partition was divided using
stratified splitting with an 80/20 ratio.

Random seed: 42.

The official test partition was kept separate.

Dataset license: CC BY-SA 4.0.
Individual image copyrights belong to their owners.

The original images are not redistributed here.

## 3. Environment

The experiments were performed on Windows using:

- Python 3.11.9
- PyTorch 2.7.1 + CUDA 11.8
- NVIDIA GeForce RTX 3060 Laptop GPU
- TorchVision
- scikit-learn
- NumPy
- Matplotlib

Exact installed package versions are provided
in requirements.txt and experiment configuration files.

## 4. Installation

Clone the repository:

```bash
git clone https://github.com/surf100/ImageClassification.git
cd ImageClassification
```

Create a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the CUDA-enabled PyTorch packages:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check the environment:

```powershell
python check_environment.py
```

CUDA-enabled GPU execution is used by the
training scripts.

## 5. Dataset Preparation

Download and explore the dataset:

```powershell
python src/explore_dataset.py
```

This downloads Oxford-IIIT Pet using TorchVision.

Create the train, validation, and test split files:

```powershell
python src/create_splits.py
```

Check preprocessing and DataLoaders:

```powershell
python src/check_data_pipeline.py
```

The project uses the official preprocessing associated
with MobileNetV3-Small DEFAULT weights.

Validation and test images use deterministic preprocessing.
Training augmentation includes horizontal flipping
for the two neural network approaches.

## 6. Model Architecture

Backbone: MobileNetV3-Small

Weights: MobileNet_V3_Small_Weights.DEFAULT
(IMAGENET1K_V1)

The final classification layer is replaced
with a Linear layer producing 37 scores.

Check the model:

```powershell
python src/check_model.py
```

## 7. Approach 1 — Full Fine-Tuning

All neural network parameters are trainable.

Configuration:

- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Learning rate: 0.0001
- Batch size: 32
- Maximum epochs: 15
- Early stopping patience: 4

Run:

```powershell
python src/train_full_finetuning.py
```

Best checkpoint selection:
Highest validation macro F1.

## 8. Approach 2 — Classifier-Head Training

The convolutional feature extractor is frozen.

The trainable layers are:

- classifier.0
- classifier.3

Configuration:

- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Learning rate: 0.001
- Batch size: 32
- Maximum epochs: 15
- Early stopping patience: 4

Run:

```powershell
python src/train_head.py
```

Best checkpoint selection:
Highest validation macro F1.

Frozen BatchNorm statistics are kept unchanged.

## 9. Approach 3 — Feature Extraction + SVM

The pretrained MobileNetV3-Small is frozen.

Feature vectors are extracted after average pooling
and before the neural network classifier.

Feature dimension: 576.

The scikit-learn pipeline uses:

- StandardScaler
- LinearSVC

Tested C values: 0.01, 0.1, 1.0, 10.0.

The scaler is fitted using training features only.
The best C is selected using validation macro F1.

Training features use deterministic preprocessing
without random augmentation.

Run:

```powershell
python src/train_svm.py
```

## 10. Final Evaluation

Evaluate the three previously selected models:

```powershell
python src/evaluate_test.py
```

IMPORTANT: This command is for reproducing the
published evaluation. It must not be used for
selecting hyperparameters or checkpoints.

The original experiments used the official test set
only for final evaluation.

## 11. Results

| Method | Test Accuracy | Macro F1 |
|---|---:|---:|
| Full Fine-Tuning | 79.89% | 0.7940 |
| Classifier-Head Training | 85.61% | 0.8519 |
| Feature Extraction + Linear SVM | 83.40% | 0.8314 |

Classifier-head training achieved the best
test performance in this experiment.

Training times and computational costs are stored
in the experiment configuration files.

For SVM, the recorded selected-pipeline time
does not include all hyperparameter-search trials.

## 12. Results and Figures

Generate comparison figures:

```powershell
python src/create_results.py
```

Generate error analysis:

```powershell
python src/analyze_errors.py
```

The results directory contains:

- Training histories and configurations
- Accuracy and macro F1 results
- Confusion matrices
- Prediction examples
- Error analysis
- Computational cost comparisons

## 13. Reproducibility

Random seeds are set for:

- Python
- NumPy
- PyTorch

The data partitions are saved in splits/.

The neural network experiments set
cudnn.deterministic=True and cudnn.benchmark=False.

Strict deterministic algorithms were not enforced.
Some variation may remain across devices and versions.

The exact selected checkpoints and hyperparameters
are recorded in results/.

The test set was not used for training,
early stopping, or hyperparameter selection.


## 14. References

Parkhi, O. M., Vedaldi, A., Zisserman, A.,
and Jawahar, C. V. (2012).
Cats and Dogs.
IEEE Conference on Computer Vision and Pattern Recognition.

Oxford-IIIT Pet Dataset:
https://www.robots.ox.ac.uk/~vgg/data/pets/

PyTorch:
https://pytorch.org/

TorchVision:
https://pytorch.org/vision/stable/
