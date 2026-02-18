
# Brain Tumor Classification using LightGBM (GPU-Accelerated)
# Dataset: Brain Tumor MRI Dataset (glioma, meningioma, notumor, pituitary)
# Model: LightGBM with GPU acceleration
# Environment: Adapted for Local Execution (originally Kaggle P100)

import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from PIL import Image
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print(f"LightGBM version: {lgb.__version__}")

# Check GPU
import subprocess
try:
    # This might fail on non-NVIDIA systems or without nvidia-smi
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout[:500])
except:
    print("No GPU detected via nvidia-smi")

# ## 1. Setup Paths & Configuration

# ============================================================
# UPDATE THIS PATH to match your local dataset
# ============================================================
DATASET_BASE = '/Users/sairammaruri/Downloads/BrainTumor_clf_TDA-main/data/Dataset'

path_train = os.path.join(DATASET_BASE, 'Training')
path_test = os.path.join(DATASET_BASE, 'Testing')
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 250

# ============================================================
# HYPERPARAMETERS
# ============================================================
PCA_VARIANCE = 0.95
NUM_ITERATIONS = 5

LGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'num_leaves': 63,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}
# ============================================================

output_dir = '/Users/sairammaruri/Downloads/BrainTumor_clf_TDA-main/models_output/LightGBM'
os.makedirs(output_dir, exist_ok=True)

# Verify paths
print("Training path exists:", os.path.exists(path_train))
print("Testing path exists:", os.path.exists(path_test))

if os.path.exists(path_train):
    for label in labels:
        p = os.path.join(path_train, label)
        count = len(os.listdir(p)) if os.path.exists(p) else 0
        print(f"  Training/{label}: {count} images")

if os.path.exists(path_test):
    for label in labels:
        p = os.path.join(path_test, label)
        count = len(os.listdir(p)) if os.path.exists(p) else 0
        print(f"  Testing/{label}: {count} images")

# ## 2. Data Loading & Preprocessing Functions

def import_data(data_path, labels, img_size):
    """Load images from folder structure."""
    images = []
    image_labels = []

    for label_idx, label in enumerate(labels):
        folder_path = os.path.join(data_path, label)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist, skipping.")
            continue

        files = sorted(os.listdir(folder_path))
        print(f"Loading {label}: {len(files)} images...")

        for filename in files:
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath).convert('L')
                img = img.resize((img_size, img_size))
                img_array = np.array(img)
                images.append(img_array)
                image_labels.append(label_idx)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    return images, image_labels


def data_to_negative(images):
    """Convert images to negative (invert pixel values)."""
    return [255 - img for img in images]

# ## 3. Load & Preprocess Data

print("=" * 60)
print("Loading Training Data...")
print("=" * 60)
x_train, y_train = import_data(path_train, labels, IMG_SIZE)

print(f"\n{'=' * 60}")
print("Loading Testing Data...")
print("=" * 60)
x_test, y_test = import_data(path_test, labels, IMG_SIZE)

print(f"\nTotal training samples: {len(x_train)}")
print(f"Total testing samples: {len(x_test)}")

# Apply negative transformation
x_train = data_to_negative(x_train)
x_test = data_to_negative(x_test)

# Normalize to [0, 1]
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

# Flatten images
X_train_flat = np.vstack([image.flatten() for image in x_train])
X_test_flat = np.vstack([image.flatten() for image in x_test])
y_train = np.array(y_train)
y_test = np.array(y_test)

del x_train, x_test

print(f"\nX_train (raw) shape: {X_train_flat.shape}")
print(f"X_test (raw) shape: {X_test_flat.shape}")

# ## 4. StandardScaler + PCA

print("=" * 60)
print("Applying StandardScaler + PCA Pipeline...")
print("=" * 60)

pca_start = time.time()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

del X_train_flat, X_test_flat

pca = PCA(n_components=PCA_VARIANCE, svd_solver='full')
X_train = pca.fit_transform(X_train_scaled)
X_test = pca.transform(X_test_scaled)

del X_train_scaled, X_test_scaled

pca_time = time.time() - pca_start

n_components = pca.n_components_
explained_var = np.sum(pca.explained_variance_ratio_) * 100

print(f"\nPCA Results:")
print(f"  Original features:    {IMG_SIZE * IMG_SIZE:,}")
print(f"  PCA components:       {n_components}")
print(f"  Variance retained:    {explained_var:.2f}%")
print(f"  Reduction:            {(1 - n_components/(IMG_SIZE*IMG_SIZE))*100:.1f}%")
print(f"  PCA fitting time:     {pca_time:.2f}s")
print(f"\nX_train (PCA) shape: {X_train.shape}")
print(f"X_test (PCA) shape:  {X_test.shape}")

# ## 5. PCA Visualization

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Cumulative explained variance
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
axes[0].plot(range(1, len(cumvar) + 1), cumvar, 'b-', linewidth=1.5)
axes[0].axhline(y=95, color='r', linestyle='--', label='95% threshold')
axes[0].axvline(x=n_components, color='g', linestyle='--', alpha=0.5, label=f'{n_components} components')
axes[0].set_xlabel('Number of Principal Components')
axes[0].set_ylabel('Cumulative Explained Variance (%)')
axes[0].set_title('PCA: Cumulative Explained Variance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D PCA projection
from sklearn.decomposition import PCA as PCA2
pca_viz = PCA2(n_components=2)
X_viz = pca_viz.fit_transform(X_train[:500])
y_viz = y_train[:500]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, label in enumerate(labels):
    mask = y_viz == i
    axes[1].scatter(X_viz[mask, 0], X_viz[mask, 1], c=colors[i],
                    label=label, alpha=0.5, s=15)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('2D PCA Projection of Training Data')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()

# ## 6. Sample Images

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, label in enumerate(labels):
    folder = os.path.join(path_train, label)
    sample_file = sorted(os.listdir(folder))[0]
    img = Image.open(os.path.join(folder, sample_file)).convert('L').resize((IMG_SIZE, IMG_SIZE))
    img_neg = 255 - np.array(img)
    axes[i].imshow(img_neg, cmap='gray')
    axes[i].set_title(f'{label}')
    axes[i].axis('off')
plt.suptitle('Sample Images (Negative Transform)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=150, bbox_inches='tight')
plt.show()

# ## 7. Train LightGBM with GPU (5 Iterations)
# 
# LightGBM natively supports GPU via `device='gpu'`.  
# Expected: ~5-15 seconds per iteration with PCA features.

model_name = "LightGBM"

# Test GPU availability - fall back to CPU if needed
try:
    test_model = lgb.LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0,
                                     n_estimators=5, verbose=-1)
    test_model.fit(X_train[:100], y_train[:100])
    USE_GPU = True
    print("GPU acceleration enabled!")
except Exception as e:
    USE_GPU = False
    print(f"GPU not available for LightGBM ({e}), falling back to CPU")
    LGBM_PARAMS['device'] = 'cpu'
    if 'gpu_platform_id' in LGBM_PARAMS:
        del LGBM_PARAMS['gpu_platform_id']
    if 'gpu_device_id' in LGBM_PARAMS:
        del LGBM_PARAMS['gpu_device_id']

metrics_funcs = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1 Score": f1_score
}

scores = {metric_name: [] for metric_name in metrics_funcs}
train_times = []
predict_times = []
last_y_pred = None

print(f"\nTraining {model_name} ({'GPU' if USE_GPU else 'CPU'}) with PCA ({n_components} components) for {NUM_ITERATIONS} iterations...")
print(f"LightGBM Config: n_estimators={LGBM_PARAMS['n_estimators']}, max_depth={LGBM_PARAMS['max_depth']}, "
      f"lr={LGBM_PARAMS['learning_rate']}, num_leaves={LGBM_PARAMS['num_leaves']}")
print("=" * 60)

for i in range(NUM_ITERATIONS):
    print(f"\n--- Iteration {i+1}/{NUM_ITERATIONS} ---")

    params = LGBM_PARAMS.copy()
    params['random_state'] = 42 + i

    model = lgb.LGBMClassifier(**params)

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict
    pred_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - pred_start

    last_y_pred = y_pred
    train_times.append(training_time)
    predict_times.append(predict_time)

    for metric_name, metric_func in metrics_funcs.items():
        if metric_name == "Accuracy":
            score = metric_func(y_test, y_pred)
        else:
            score = metric_func(y_test, y_pred, average='weighted')
        scores[metric_name].append(score)

    print(f"  Training time:   {training_time:.2f}s")
    print(f"  Prediction time: {predict_time:.2f}s")
    print(f"  Accuracy:        {scores['Accuracy'][-1]:.4f}")
    print(f"  F1 Score:        {scores['F1 Score'][-1]:.4f}")

total_train_time = sum(train_times)
print(f"\n{'=' * 60}")
print(f"Training Complete! Total time: {total_train_time:.2f}s ({total_train_time/60:.1f} min)")
print("=" * 60)

# ## 8. Results & Statistics

avg_scores = {k: np.mean(v) for k, v in scores.items()}
std_scores = {k: np.std(v) for k, v in scores.items()}
avg_time = np.mean(train_times)
std_time = np.std(train_times)
avg_pred_time = np.mean(predict_times)

print(f"\n{'=' * 60}")
print(f"Average Performance of {model_name} + PCA ({NUM_ITERATIONS} runs)")
print(f"{'=' * 60}")
for metric_name, avg_score in avg_scores.items():
    std_score = std_scores[metric_name]
    print(f"  {metric_name}: {avg_score:.4f} ± {std_score:.4f}")
print(f"  Training Time:   {avg_time:.2f}s ± {std_time:.2f}s")
print(f"  Prediction Time: {avg_pred_time:.2f}s")
print(f"  PCA Time:        {pca_time:.2f}s (one-time)")
print(f"  GPU Used:        {USE_GPU}")

print(f"\n{'=' * 60}")
print("Classification Report (Last Iteration)")
print(f"{'=' * 60}")
print(classification_report(y_test, last_y_pred, target_names=labels))

# ## 9. Confusion Matrix

cm = confusion_matrix(y_test, last_y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', ax=ax)
plt.title(f"{model_name} ({'GPU' if USE_GPU else 'CPU'}) + PCA ({n_components} components) - Confusion Matrix", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Confusion matrix saved to {cm_path}")

# ## 10. Feature Importance (Top 30 PCA Components)

importances = model.feature_importances_
top_n = min(30, len(importances))
top_idx = np.argsort(importances)[-top_n:]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(top_n), importances[top_idx], color='steelblue', alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels([f'PC{idx+1}' for idx in top_idx])
ax.set_xlabel('Feature Importance (Split Count)')
ax.set_title(f'Top {top_n} Most Important PCA Components (LightGBM)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

# ## 11. Performance Across Iterations

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

iterations = range(1, NUM_ITERATIONS + 1)
for metric_name, score_list in scores.items():
    axes[0].plot(iterations, score_list, 'o-', label=metric_name, linewidth=2, markersize=8)
axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Metrics Across Iterations', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, alpha=0.3)

axes[1].bar(iterations, train_times, color='steelblue', alpha=0.8, label='Training')
axes[1].bar(iterations, predict_times, bottom=train_times, color='coral', alpha=0.8, label='Prediction')
axes[1].axhline(y=avg_time, color='red', linestyle='--', label=f'Mean Train: {avg_time:.2f}s')
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Time (seconds)', fontsize=12)
axes[1].set_title('Training & Prediction Time per Iteration', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_plots.png'), dpi=150, bbox_inches='tight')
plt.show()

# ## 12. Save Metrics to JSON

results = {
    "model": model_name,
    "gpu_accelerated": USE_GPU,
    "pipeline": f"Grayscale → Negative → Normalize → StandardScaler → PCA → LightGBM ({'GPU' if USE_GPU else 'CPU'})",
    "num_iterations": NUM_ITERATIONS,
    "image_size": IMG_SIZE,
    "original_features": IMG_SIZE * IMG_SIZE,
    "pca_components": int(n_components),
    "pca_variance_retained": f"{explained_var:.2f}%",
    "pca_time_seconds": round(pca_time, 2),
    "dimensionality_reduction": f"{IMG_SIZE*IMG_SIZE} → {n_components} ({(1 - n_components/(IMG_SIZE*IMG_SIZE))*100:.1f}% reduction)",
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "class_distribution": {
        "train": {label: int(np.sum(y_train == i)) for i, label in enumerate(labels)},
        "test": {label: int(np.sum(y_test == i)) for i, label in enumerate(labels)}
    },
    "average_metrics": {
        k: f"{v:.4f} ± {std_scores[k]:.4f}" for k, v in avg_scores.items()
    },
    "raw_scores": {k: [round(s, 4) for s in v] for k, v in scores.items()},
    "training_times": [round(t, 2) for t in train_times],
    "prediction_times": [round(t, 2) for t in predict_times],
    "training_time_avg": f"{avg_time:.2f} ± {std_time:.2f}",
    "total_training_time": f"{total_train_time:.2f}s ({total_train_time/60:.1f} min)",
    "lgbm_params": {k: str(v) for k, v in LGBM_PARAMS.items()}
}

metrics_path = os.path.join(output_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Metrics saved to {metrics_path}")
print("\nFull results:")
print(json.dumps(results, indent=4))

# ## 13. Download Outputs
# All outputs saved in output_dir

print("\nOutput files:")
for f in sorted(os.listdir(output_dir)):
    filepath = os.path.join(output_dir, f)
    # Check if it is a file before getting size
    if os.path.isfile(filepath):
        size = os.path.getsize(filepath) / 1024
        print(f"  {f} ({size:.1f} KB)")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Pipeline:     Image → Negative → Scale → PCA({n_components}) → LightGBM({'GPU' if USE_GPU else 'CPU'})")
print(f"  GPU Used:     {USE_GPU}")
print(f"  Accuracy:     {avg_scores['Accuracy']:.4f} ± {std_scores['Accuracy']:.4f}")
print(f"  F1 Score:     {avg_scores['F1 Score']:.4f} ± {std_scores['F1 Score']:.4f}")
print(f"  Train Time:   {avg_time:.2f}s per iteration")
print(f"  Total Time:   {total_train_time:.2f}s ({total_train_time/60:.1f} min)")
print(f"{'=' * 60}")
