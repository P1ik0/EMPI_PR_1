import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, roc_curve, roc_auc_score

df = pd.read_csv('data_metrics.csv')
print(df.head())

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
print(df.head())

print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))


def demchenko_find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def demchenko_find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def demchenko_find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def demchenko_find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


print('TP:', demchenko_find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', demchenko_find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', demchenko_find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', demchenko_find_TN(df.actual_label.values, df.predicted_RF.values))


def demchenko_find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = demchenko_find_TP(y_true, y_pred)
    FN = demchenko_find_FN(y_true, y_pred)
    FP = demchenko_find_FP(y_true, y_pred)
    TN = demchenko_find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def demchenko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = demchenko_find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


print(demchenko_confusion_matrix(df.actual_label.values, df.predicted_RF.values))

assert np.array_equal(demchenko_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
    'demchenko_confusion_matrix() is not correct for RF'
assert np.array_equal(demchenko_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
    'demchenko_confusion_matrix() is not correct for LR'

print(accuracy_score(df.actual_label.values, df.predicted_RF.values))


def demchenko_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = demchenko_find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)


assert demchenko_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(
    df.actual_label.values, df.predicted_RF.values), 'demchenko_accuracy_score failed on RF'
assert demchenko_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(
    df.actual_label.values, df.predicted_LR.values), 'demchenko_accuracy_score failed on LR'
print('Accuracy RF: %.3f' % (demchenko_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (demchenko_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

print(recall_score(df.actual_label.values, df.predicted_RF.values))


def demchenko_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP, FN, FP, TN = demchenko_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


assert demchenko_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(
    df.actual_label.values, df.predicted_RF.values), 'demchenko_recall_score failed on RF'
assert demchenko_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(
    df.actual_label.values, df.predicted_LR.values), 'demchenko_recall_score failed on LR'
print('Recall RF: %.3f' % (demchenko_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (demchenko_recall_score(df.actual_label.values, df.predicted_LR.values)))

print(precision_score(df.actual_label.values, df.predicted_RF.values))


def demchenko_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    TP, FN, FP, TN = demchenko_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


assert demchenko_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(
    df.actual_label.values, df.predicted_RF.values), 'demchenko_precision_score failed on RF'
assert demchenko_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values, df.predicted_LR.values), 'demchenko_precision_score failed on LR'
print('Precision RF: %.3f' % (demchenko_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (demchenko_precision_score(df.actual_label.values, df.predicted_LR.values)))

print(f1_score(df.actual_label.values, df.predicted_RF.values))


def demchenko_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = demchenko_recall_score(y_true, y_pred)
    precision = demchenko_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


assert np.isclose(demchenko_f1_score(df.actual_label.values, df.predicted_RF.values),
                  f1_score(df.actual_label.values, df.predicted_RF.values)), 'demchenko_f1_score failed on RF'
assert np.isclose(demchenko_f1_score(df.actual_label.values, df.predicted_LR.values),
                  f1_score(df.actual_label.values, df.predicted_LR.values)), 'demchenko_f1_score failed on LR'
print('F1 RF: %.3f' % (demchenko_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (demchenko_f1_score(df.actual_label.values, df.predicted_LR.values)))

print('')
print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (demchenko_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (demchenko_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (demchenko_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (demchenko_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (demchenko_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (demchenko_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (demchenko_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (demchenko_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
