import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, accuracy_score,
    matthews_corrcoef, confusion_matrix
)

X = np.loadtxt('../features/SampleFeature_best.csv', delimiter=',')
y = np.concatenate((np.ones(len(X) // 2), np.zeros(len(X) // 2)))

clf = MLPClassifier(
    learning_rate_init=0.5,
    max_iter=500,
    random_state=90
)

skf = KFold(n_splits=5, shuffle=True, random_state=80)

precision_list, recall_list, f1_score_list = [], [], []
acc_list, mcc_list, spec_list = [], [], []
aucs, pr_aucs = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    _, _, _ = precision_recall_curve(y_test, y_pred_prob)
    aupr = average_precision_score(y_test, y_pred_prob)
    pr_aucs.append(aupr)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)

    precision_list.append(prec)
    recall_list.append(rec)
    f1_score_list.append(f1)
    acc_list.append(acc)
    mcc_list.append(mcc)
    spec_list.append(spec)

    print(f"Fold {fold+1}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Specificity: {spec:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

print("\n====== 5-Fold Mean Metrics ======")
print(f"Avg Accuracy:     {np.mean(acc_list):.4f}")
print(f"Avg Precision:    {np.mean(precision_list):.4f}")
print(f"Avg Recall:       {np.mean(recall_list):.4f}")
print(f"Avg Specificity:  {np.mean(spec_list):.4f}")
print(f"Avg F1-Score:     {np.mean(f1_score_list):.4f}")
print(f"Avg MCC:          {np.mean(mcc_list):.4f}")
print(f"Avg AUC:          {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
print(f"Avg AUPR:         {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs, ddof=1):.4f}")
