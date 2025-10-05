import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype(int)
df['predicted_LR'] = (df.model_LR >= thresh).astype(int)

def find_TP(y_true, y_pred):
    return int(((y_true == 1) & (y_pred == 1)).sum())

def find_FN(y_true, y_pred):
    return int(((y_true == 1) & (y_pred == 0)).sum())

def find_FP(y_true, y_pred):
    return int(((y_true == 0) & (y_pred == 1)).sum())

def find_TN(y_true, y_pred):
    return int(((y_true == 0) & (y_pred == 0)).sum())

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def lavreniuk_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

assert np.array_equal(
    lavreniuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
    confusion_matrix(df.actual_label.values, df.predicted_RF.values)
)

assert np.array_equal(
    lavreniuk_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
    confusion_matrix(df.actual_label.values, df.predicted_LR.values)
)

print("TP/FN/FP/TN (RF):",
      find_TP(df.actual_label.values, df.predicted_RF.values),
      find_FN(df.actual_label.values, df.predicted_RF.values),
      find_FP(df.actual_label.values, df.predicted_RF.values),
      find_TN(df.actual_label.values, df.predicted_RF.values))


def lavreniuk_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def lavreniuk_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return 0.0 if (TP + FN) == 0 else TP / (TP + FN)

def lavreniuk_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return 0.0 if (TP + FP) == 0 else TP / (TP + FP)

def lavreniuk_f1_score(y_true, y_pred):
    r = lavreniuk_recall_score(y_true, y_pred)
    p = lavreniuk_precision_score(y_true, y_pred)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


for col in ['predicted_RF', 'predicted_LR']:
    y_true = df.actual_label.values
    y_pred = df[col].values
    assert abs(lavreniuk_accuracy_score(y_true, y_pred) - accuracy_score(y_true, y_pred)) < 1e-12
    assert abs(lavreniuk_recall_score(y_true, y_pred) - recall_score(y_true, y_pred)) < 1e-12
    assert abs(lavreniuk_precision_score(y_true, y_pred) - precision_score(y_true, y_pred)) < 1e-12
    assert abs(lavreniuk_f1_score(y_true, y_pred) - f1_score(y_true, y_pred)) < 1e-12

print(f"Accuracy RF: {lavreniuk_accuracy_score(df.actual_label.values, df.predicted_RF.values):.3f}")
print(f"Accuracy LR: {lavreniuk_accuracy_score(df.actual_label.values, df.predicted_LR.values):.3f}")
print(f"Recall   RF: {lavreniuk_recall_score(df.actual_label.values, df.predicted_RF.values):.3f}")
print(f"Recall   LR: {lavreniuk_recall_score(df.actual_label.values, df.predicted_LR.values):.3f}")
print(f"Precision RF: {lavreniuk_precision_score(df.actual_label.values, df.predicted_RF.values):.3f}")
print(f"Precision LR: {lavreniuk_precision_score(df.actual_label.values, df.predicted_LR.values):.3f}")
print(f"F1        RF: {lavreniuk_f1_score(df.actual_label.values, df.predicted_RF.values):.3f}")
print(f"F1        LR: {lavreniuk_f1_score(df.actual_label.values, df.predicted_LR.values):.3f}")

print("\nScores with threshold = 0.50 (RF)")
y_pred_050 = (df.model_RF.values >= 0.50).astype(int)
print(f"Accuracy:  {lavreniuk_accuracy_score(df.actual_label.values, y_pred_050):.3f}")
print(f"Recall:    {lavreniuk_recall_score(df.actual_label.values, y_pred_050):.3f}")
print(f"Precision: {lavreniuk_precision_score(df.actual_label.values, y_pred_050):.3f}")
print(f"F1:        {lavreniuk_f1_score(df.actual_label.values, y_pred_050):.3f}")

print("\nScores with threshold = 0.25 (RF)")
y_pred_025 = (df.model_RF.values >= 0.25).astype(int)
print(f"Accuracy:  {lavreniuk_accuracy_score(df.actual_label.values, y_pred_025):.3f}")
print(f"Recall:    {lavreniuk_recall_score(df.actual_label.values, y_pred_025):.3f}")
print(f"Precision: {lavreniuk_precision_score(df.actual_label.values, y_pred_025):.3f}")
print(f"F1:        {lavreniuk_f1_score(df.actual_label.values, y_pred_025):.3f}")

fpr_RF, tpr_RF, thr_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thr_LR = roc_curve(df.actual_label.values, df.model_LR.values)
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

plt.figure(figsize=(6,5))
plt.plot(fpr_RF, tpr_RF, label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0,1],[0,1], 'k--', label='random')
plt.plot([0,0,1,1],[0,1,1,1], 'g--', label='perfect')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (RF vs LR)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/roc_rf_vs_lr.png', dpi=150)
plt.close()
print("Saved ROC figure to outputs/roc_rf_vs_lr.png")