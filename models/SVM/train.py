# # Brain Tumor Classification using SVM with PCA Optimization
# **Dataset:** Brain Tumor MRI Dataset (glioma, meningioma, notumor, pituitary)  
# **Model:** Support Vector Machine (SVM) with PCA Dimensionality Reduction  
# **Environment:** Kaggle P100 (SVM uses CPU; PCA reduces dimensionality for faster training)
# 
# ### Why PCA + SVM?
# - Raw images at 250×250 = 62,500 features → extremely slow SVM training
# - PCA reduces to ~500 components while retaining 95%+ variance
# - Training time drops from ~2 hours to ~5-10 minutes (5 iterations)
# - Accuracy remains comparable or even improves (reduces noise/overfitting)
# - PCA + SVM is a well-established, publishable ML pipeline

import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ## 1. Setup Paths & Configuration


DATASET_BASE = '/Users/sairammaruri/Downloads/BrainTumor_clf_TDA-main/data/Dataset'
path_train = os.path.join(DATASET_BASE, 'Training')
path_test = os.path.join(DATASET_BASE, 'Testing')
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 250

# ============================================================
# KEY HYPERPARAMETERS - Tune these for your paper
# ============================================================
PCA_VARIANCE = 0.95          # Retain 95% of variance (paper-standard)
SVM_KERNEL = 'rbf'           # RBF kernel
SVM_C = 1.0                  # Regularization parameter
SVM_GAMMA = 'scale'          # Gamma parameter
NUM_ITERATIONS = 5           # Number of training runs
# ============================================================

output_dir = '/Users/sairammaruri/Downloads/BrainTumor_clf_TDA-main/models_output/SVM'
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
                img = Image.open(filepath).convert('L')  # Grayscale
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

# Free memory
del x_train, x_test

print(f"\nX_train (raw) shape: {X_train_flat.shape}")
print(f"X_test (raw) shape: {X_test_flat.shape}")

# ## 4. Feature Extraction Pipeline: StandardScaler + PCA
# 
# **Why this matters for your paper:**
# - StandardScaler normalizes features to zero mean, unit variance → essential before PCA
# - PCA with 95% variance retention is a standard, well-justified choice
# - Reduces 62,500 features to ~300-500 principal components
# - Dramatically faster SVM training with minimal information loss

print("=" * 60)
print("Applying StandardScaler + PCA Pipeline...")
print("=" * 60)

# Step 1: StandardScaler (fit on train, transform both)
pca_start = time.time()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Free raw flat data
del X_train_flat, X_test_flat

# Step 2: PCA with 95% variance retention
pca = PCA(n_components=PCA_VARIANCE, svd_solver='full')
X_train = pca.fit_transform(X_train_scaled)
X_test = pca.transform(X_test_scaled)

# Free scaled data
del X_train_scaled, X_test_scaled

pca_time = time.time() - pca_start

n_components = pca.n_components_
explained_var = np.sum(pca.explained_variance_ratio_) * 100

print(f"\nPCA Results:")
print(f"  Original features:    {IMG_SIZE * IMG_SIZE:,} (250×250)")
print(f"  PCA components:       {n_components}")
print(f"  Variance retained:    {explained_var:.2f}%")
print(f"  Dimensionality reduction: {IMG_SIZE * IMG_SIZE:,} → {n_components} ({(1 - n_components/(IMG_SIZE*IMG_SIZE))*100:.1f}% reduction)")
print(f"  PCA fitting time:     {pca_time:.2f}s")
print(f"\nX_train (PCA) shape: {X_train.shape}")
print(f"X_test (PCA) shape:  {X_test.shape}")

# ## 5. Visualize PCA: Explained Variance & 2D Projection

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

# 2D PCA projection for visualization
pca_2d = PCA(n_components=2)
X_train_2d = pca_2d.fit_transform(scaler.fit_transform(
    np.vstack([Image.open(os.path.join(path_train, labels[int(y)], 
    sorted(os.listdir(os.path.join(path_train, labels[int(y)])))[0])).convert('L').resize((IMG_SIZE, IMG_SIZE))
    for y in y_train[:200]]).reshape(200, -1) / 255.0
) if False else X_train)  # Use already-transformed data
X_2d_sample = X_train[:500]
y_2d_sample = y_train[:500]

# Quick 2D PCA for visualization
from sklearn.decomposition import PCA as PCA2
pca_viz = PCA2(n_components=2)
X_viz = pca_viz.fit_transform(X_train[:500])

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, label in enumerate(labels):
    mask = y_2d_sample == i
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
print("PCA analysis plot saved.")

# ## 6. Visualize Sample Images

# Reload a few samples just for visualization
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

# ## 7. Train SVM (5 Iterations)
# 
# With PCA reducing features from 62,500 → ~300-500:
# - **Before PCA:** ~20-30 min per iteration → ~2+ hours total
# - **After PCA:** ~1-2 min per iteration → ~5-10 minutes total

model_name = "SVM"

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

print(f"Training {model_name} with PCA ({n_components} components) for {NUM_ITERATIONS} iterations...")
print(f"SVM Config: kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA}")
print("=" * 60)

for i in range(NUM_ITERATIONS):
    print(f"\n--- Iteration {i+1}/{NUM_ITERATIONS} ---")

    model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA)

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

    # Calculate metrics
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

print(f"\n{'=' * 60}")
print("Classification Report (Last Iteration)")
print(f"{'=' * 60}")
print(classification_report(y_test, last_y_pred, target_names=labels))

# ## 9. Confusion Matrix

cm = confusion_matrix(y_test, last_y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', ax=ax)
plt.title(f"{model_name} + PCA ({n_components} components) - Confusion Matrix", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Confusion matrix saved to {cm_path}")

# ## 10. Performance Across Iterations

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

# ## 11. Save All Metrics to JSON

results = {
    "model": model_name,
    "pipeline": "Grayscale → Negative → Normalize → StandardScaler → PCA → SVM (RBF)",
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
    "svm_params": {
        "kernel": SVM_KERNEL,
        "C": SVM_C,
        "gamma": SVM_GAMMA
    }
}

metrics_path = os.path.join(output_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Metrics saved to {metrics_path}")
print("\nFull results:")
print(json.dumps(results, indent=4))

# ## 12. Download Outputs
# All outputs are saved in `/Users/sairammaruri/Downloads/BrainTumor_clf_TDA-main/models_output/SVM_PCA_output/`:
# - `confusion_matrix.png`
# - `sample_images.png`  
# - `pca_analysis.png`
# - `performance_plots.png`
# - `metrics.json`

print("\nOutput files:")
for f in sorted(os.listdir(output_dir)):
    filepath = os.path.join(output_dir, f)
    size = os.path.getsize(filepath) / 1024
    print(f"  {f} ({size:.1f} KB)")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Pipeline:     Image → Negative → Scale → PCA({n_components}) → SVM(RBF)")
print(f"  Accuracy:     {avg_scores['Accuracy']:.4f} ± {std_scores['Accuracy']:.4f}")
print(f"  F1 Score:     {avg_scores['F1 Score']:.4f} ± {std_scores['F1 Score']:.4f}")
print(f"  Train Time:   {avg_time:.2f}s per iteration")
print(f"  Total Time:   {total_train_time:.2f}s ({total_train_time/60:.1f} min)")
print(f"{'=' * 60}")