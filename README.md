# Brain Tumor Classification using TDA and ML

A machine learning pipeline for classifying brain tumor MRI images into four categories using PCA-based dimensionality reduction and multiple classical ML classifiers.

## Dataset

The project uses the **Brain Tumor MRI Dataset** with four classes:

| Split    | Glioma | Meningioma | No Tumor | Pituitary | Total |
|----------|--------|------------|----------|-----------|-------|
| Training | 1,321  | 1,339      | 1,595    | 1,457     | 5,712 |
| Testing  | 300    | 306        | 405      | 300       | 1,311 |

Images are 250x250 grayscale MRI scans.

## Environment

- **Training Platform**: Kaggle (NVIDIA P100 GPU)
- GPU-accelerated models: XGBoost (`gpu_hist`), LightGBM (`device='gpu'`)
- CPU-based models: SVM, KNN, Random Forest, Extra Trees, AdaBoost

## Pipeline

```
Grayscale → Negative Transform → Normalize [0,1] → StandardScaler → PCA (95% variance) → Classifier
```

- **Negative Transform**: Inverts pixel values (255 - pixel) to enhance tumor regions
- **PCA**: Reduces 62,500 features to ~1,031 principal components while retaining 95% variance (~98.4% dimensionality reduction)

## Models & Results

All models were trained for 5 iterations to measure variance. Results averaged across runs:

| Model | Accuracy | F1 Score | Avg Train Time |
|-------|----------|----------|----------------|
| **KNN** (k=5, distance-weighted) | **93.75%** | **93.67%** | 0.02s |
| **LightGBM** (GPU) | **93.30%** | **93.24%** | 46.42s |
| **XGBoost** | **92.75%** | **92.68%** | 6.63s |
| **Extra Trees** (300 trees) | **91.41%** | **91.36%** | 4.32s |
| **Random Forest** (300 trees) | **90.47%** | **90.35%** | 31.36s |
| **SVM** (RBF kernel) | **90.08%** | **89.98%** | 5.96s |
| **AdaBoost** (SAMME) | **85.74%** | **85.58%** | 608.41s |

## Project Structure

```
├── Dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── models/
│   ├── AdaBoost/train.py
│   ├── Extra_Trees/train.py
│   ├── KNN/train.py
│   ├── LightGBM/train.py
│   ├── Random_Forest/train.py
│   ├── SVM/train.py
│   └── XGBoost/train.py
├── models_output/           # Saved metrics JSON + plots per model
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Each model has its own training script under `models/<ModelName>/train.py`. To train a specific model:

```bash
python models/KNN/train.py
python models/SVM/train.py
python models/Random_Forest/train.py
python models/XGBoost/train.py
python models/LightGBM/train.py
python models/Extra_Trees/train.py
python models/AdaBoost/train.py
```

Each script will:
1. Load and preprocess the dataset
2. Apply StandardScaler + PCA
3. Train the model for 5 iterations
4. Output metrics (Accuracy, Precision, Recall, F1), confusion matrix, and plots to `models_output/<ModelName>/`

## Output per Model

Each model generates the following in `models_output/<ModelName>/`:
- `metrics.json` - Full performance metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `pca_analysis.png` - PCA variance and 2D projection plots
- `sample_images.png` - Sample negative-transformed images
- `feature_importance.png` - Top PCA component importances (tree-based models)
- `performance_plots.png` - Metrics and timing across iterations

## Author

**Sairam Maruri** - [saiii.in](https://saiii.in)
