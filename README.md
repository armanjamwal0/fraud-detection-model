# Fraud Detection Model

A machine learning pipeline for detecting fraudulent transactions in a large-scale, highly imbalanced dataset.

## Overview

This project builds a fraud detection model trained on a real-world-scale transactional dataset. The core challenges were **working with a large dataset (10M+ rows)** under limited compute, and **handling severe class imbalance** (fraud cases make up only ~4% of transactions) — both of which are common, hard problems in real fraud detection systems.

## Dataset

| Property | Value |
|---|---|
| Original size | 10,000,000 instances |
| Features | 23 |
| Sampled size (for training) | 200,000 instances |
| Class distribution | 96% legitimate, 4% fraud |
| Sampling method | Stratified sampling via `train_test_split` (preserves class ratio)|

The full dataset was too large to work with efficiently, so it was **downsampled to 200,000 rows** using stratified sampling, which preserves the original 96/4 class distribution in the sampled subset. The same stratification approach was then reused to split the sampled data into train/test sets, so the class ratio stays consistent throughout the pipeline.
 
```python
# Step 1: Downsample the full dataset to a smaller subset, preserving class ratio
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, stratify=y)
 
# Step 2: Split the sampled data into train/test sets, again preserving class ratio
X_train, x_test, Y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
)
```
 
Using `stratify=y` in both steps ensures the fraud/legitimate ratio (4%/96%) is maintained in every subset — the sample, the training set, and the test set — so the imbalance problem isn't accidentally worsened or hidden by an unlucky random split.

## Challenges Faced

1. **Loading and handling a 10M-row dataset** — required an efficient sampling strategy to reduce the dataset to a workable size without losing representativeness.
2. **Severe class imbalance (96/4 split)** — a naive model trained on this data would default to predicting "legitimate" almost always, achieving high accuracy while being useless at catching fraud .

## Methodology

### 1. Data Sampling
Reduced 10M rows → 200K rows using [Add: sampling technique], maintaining the original 96/4 class ratio.

### 2. Handling Class Imbalance
Explored multiple strategies for imbalanced classification:
- **Cost-sensitive weighting** (final approach used) — penalizes misclassification of the minority (fraud) class more heavily during training, without altering the dataset itself
- **SMOTE** (Synthetic Minority Over-sampling Technique) — researched as an alternative that generates synthetic minority-class samples
- **Balanced Random Forest** — researched as an alternative ensemble method with built-in class balancing

Cost-sensitive weighting was selected as the primary method for this iteration.

### 3. Preprocessing



 - **Missing value imputation** — filled missing values to ensure a complete dataset for training
- **Feature scaling** — numerical features scaled to bring them onto a comparable range
- **Feature encoding** — categorical features encoded into a numerical format usable by the model




### 4. Model Training
- Model: [ balancedRandomForest , Random Forest ,  XGBoost]
- Class weighting applied during training to counteract imbalance

### 5. Evaluation
Model performance was evaluated using metrics appropriate for imbalanced classification (accuracy alone would be misleading here):
- **Confusion Matrix** — to see the breakdown of true/false positives and negatives, not just an aggregate score
- **ROC Curve** — to evaluate how well the model separates the two classes across thresholds
- **Recall** — prioritized as the key metric, since missing actual fraud cases (false negatives) is more costly than flagging a legitimate transaction for review (false positive)

### 6. Overfitting / Underfitting Check
Training and validation performance were compared to confirm the model generalizes rather than memorizing the training set. 

## Results

| Metric | Before Fix | After Fix |
|---|---|---|
| Recall (fraud class) | 0.11 | 0.69 |
| F1 Score | 0.35 | 0.81 
| AUC |    | 0.71

The initial model, without addressing class imbalance, achieved a recall of only **0.11** — meaning it missed the vast majority of actual fraud cases despite likely having high overall accuracy. After applying cost-sensitive weighting, recall improved to **0.69**, a substantial gain in the model's ability to correctly identify fraudulent transactions.


## Key Learnings

- **Accuracy is a misleading metric on imbalanced datasets** — a model can score 96% accuracy by always predicting "legitimate" while being completely useless for fraud detection. Recall (and precision/F1) matter far more here.
- **Cost-sensitive learning** is an effective way to handle imbalance without modifying the dataset, by making the model "pay more attention" to the minority class during training.
- Explored and compared alternative imbalance-handling techniques (**SMOTE**, **Balanced Random Forest**) as part of the research process, even though cost-weighting was the final choice.
- Sampling large datasets requires care to preserve the underlying class distribution — otherwise the imbalance problem gets distorted before modeling even begins.


# fraud-detection-model
