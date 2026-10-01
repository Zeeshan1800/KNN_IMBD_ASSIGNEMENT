import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

import time
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, accuracy_score)

from sklearn.inspection import permutation_importance



# Load the Kaggle IMDB 5000 dataset
df = pd.read_csv('movie_metadata.csv')

# Basic inspection
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum().sort_values(ascending=False).head(15))

# Create binary classification target
# High-rated: imdb_score >= 7.0 (1), Low-rated: < 7.0 (0)
df = df.dropna(subset=['imdb_score'])
df['high_rated'] = (df['imdb_score'] >= 7.0).astype(int)

print("Class distribution:")
print(df['high_rated'].value_counts())
print("\nProportion:")
print(df['high_rated'].value_counts(normalize=True))

# Target distribution
sns.countplot(x='high_rated', data=df)
plt.title('High-Rated vs Low-Rated Movies')
plt.show()

# Distribution of key numerical features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ['budget', 'gross', 'duration', 'num_voted_users', 
            'num_critic_for_reviews', 'director_facebook_likes']
for ax, feat in zip(axes.flatten(), features):
    df[feat].hist(ax=ax, bins=30)
    ax.set_title(feat)
plt.tight_layout()
plt.show()

# Correlation heatmap (numerical only)
numerical_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(14, 10))
sns.heatmap(df[numerical_cols].corr(), cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Boxplots: key features vs target
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(x='high_rated', y=feat, data=df, ax=ax)
    ax.set_title(f'{feat} by Rating Class')
plt.tight_layout()
plt.show()

# Select relevant features (drop high-cardinality text columns for KNN)
features = [
    'budget', 'gross', 'duration', 'num_voted_users',
    'num_critic_for_reviews', 'num_user_for_reviews',
    'director_facebook_likes', 'actor_1_facebook_likes',
    'actor_2_facebook_likes', 'actor_3_facebook_likes',
    'cast_total_facebook_likes', 'facenumber_in_poster',
    'title_year', 'aspect_ratio'
]

# Handle missing values
df_model = df[features + ['high_rated']].copy()

# Drop rows where critical financial features are missing or zero
df_model = df_model[(df_model['budget'] > 0) & (df_model['gross'] > 0)]

# Impute remaining missing values with median
for col in df_model.columns:
    if df_model[col].isnull().sum() > 0:
        df_model[col].fillna(df_model[col].median(), inplace=True)

print("After cleaning:", df_model.shape)

# Feature engineering: create additional derived features
df_model['budget_gross_ratio'] = df_model['gross'] / df_model['budget']
df_model['votes_per_year'] = df_model['num_voted_users'] / (2026 - df_model['title_year'] + 1)
df_model['critic_user_ratio'] = df_model['num_critic_for_reviews'] / (df_model['num_user_for_reviews'] + 1)

# Check for infinity from division
df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
df_model.fillna(df_model.median(), inplace=True)

# Separate features and target
X = df_model.drop('high_rated', axis=1)
y = df_model['high_rated']

# Feature scaling — CRITICAL for KNN (distance-based)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Final feature set shape:", X_scaled.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    stratify=y,          # Preserves class proportions
    random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train class balance:", y_train.mean())
print("Test class balance:", y_test.mean())





k_values = range(1, 16)
cv_scores = []
error_rates = []

# Use StratifiedKFold for imbalanced classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='accuracy')
    cv_scores.append(scores.mean())
    error_rates.append(1 - scores.mean())

# Plot both curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_values, cv_scores, marker='o', color='blue', linewidth=2)
axes[0].set_xlabel('K (Number of Neighbors)')
axes[0].set_ylabel('CV Accuracy')
axes[0].set_title('K vs Cross-Validation Score')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_values, error_rates, marker='o', color='red', linewidth=2)
axes[1].set_xlabel('K (Number of Neighbors)')
axes[1].set_ylabel('Error Rate')
axes[1].set_title('Elbow Curve (Error Rate vs K)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Identify optimal K
best_k = k_values[np.argmax(cv_scores)]
print(f"Best K by CV Score: {best_k}")
print(f"Best CV Accuracy: {max(cv_scores):.4f}")




# Train final model with optimal K
final_knn = KNeighborsClassifier(n_neighbors=best_k)

# Measure training time
t0 = time.time()
final_knn.fit(X_train, y_train)
train_time = time.time() - t0

# Measure prediction time
t0 = time.time()
y_pred = final_knn.predict(X_test)
y_proba = final_knn.predict_proba(X_test)[:, 1]
pred_time = time.time() - t0

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low-Rated', 'High-Rated'],
            yticklabels=['Low-Rated', 'High-Rated'])
plt.title(f'Confusion Matrix (K={best_k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low-Rated', 'High-Rated']))

# ROC-AUC
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Performance Summary
train_acc = final_knn.score(X_train, y_train)
test_acc = accuracy_score(y_test, y_pred)

print("=" * 50)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 50)
print(f"Optimal K         : {best_k}")
print(f"Train Accuracy    : {train_acc:.4f}")
print(f"Test Accuracy     : {test_acc:.4f}")
print(f"ROC-AUC Score     : {auc:.4f}")
print(f"Training Time     : {train_time:.4f} seconds")
print(f"Prediction Time   : {pred_time:.4f} seconds")
print(f"Overfitting Gap   : {train_acc - test_acc:.4f}")
print("=" * 50)




result = permutation_importance(final_knn, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)

print("Top 10 Features by Permutation Importance:")
print(importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df.head(10))
plt.title('Feature Importance (Permutation-based)')
plt.tight_layout()
plt.show()