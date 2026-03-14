# tuning.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score, classification_report

def run_tuning(X, y, contamination_values=None):
    if contamination_values is None:
        contamination_values = [0.0005, 0.001, 0.0015, 0.00173, 0.002, 0.0025, 0.003]
    
    print("="*60)
    print("HYPERPARAMETER TUNING – Contamination / nu grid")
    print("="*60)
    
    tuning_results = []
    
    for cont in contamination_values:
        print(f"\nTesting contamination / nu = {cont:.5f}")

        iso = IsolationForest(contamination=cont, random_state=42, n_estimators=100)
        iso_anom = (iso.fit_predict(X) == -1).astype(int)
        pr_iso = average_precision_score(y, iso_anom)
        rec_iso = classification_report(y, iso_anom, output_dict=True, zero_division=0)['1']['recall']
        prec_iso = classification_report(y, iso_anom, output_dict=True, zero_division=0)['1']['precision']
  
        lof = LocalOutlierFactor(n_neighbors=20, contamination=cont, novelty=True)
        lof.fit(X)
        lof_anom = (lof.predict(X) == -1).astype(int)
        pr_lof = average_precision_score(y, lof_anom)
        rec_lof = classification_report(y, lof_anom, output_dict=True, zero_division=0)['1']['recall']
        prec_lof = classification_report(y, lof_anom, output_dict=True, zero_division=0)['1']['precision']
        
        svm = OneClassSVM(kernel='rbf', nu=cont)
        svm_anom = (svm.fit_predict(X) == -1).astype(int)
        pr_svm = average_precision_score(y, svm_anom)
        rec_svm = classification_report(y, svm_anom, output_dict=True, zero_division=0)['1']['recall']
        prec_svm = classification_report(y, svm_anom, output_dict=True, zero_division=0)['1']['precision']
        
        tuning_results.append({
            'contamination': cont,
            'ISO_PR-AUC': round(pr_iso, 4),
            'ISO_Recall': round(rec_iso, 4),
            'ISO_Prec': round(prec_iso, 4),
            'LOF_PR-AUC': round(pr_lof, 4),
            'LOF_Recall': round(rec_lof, 4),
            'LOF_Prec': round(prec_lof, 4),
            'SVM_PR-AUC': round(pr_svm, 4),
            'SVM_Recall': round(rec_svm, 4),
            'SVM_Prec': round(prec_svm, 4),
        })
    
    tuning_df = pd.DataFrame(tuning_results)
    print("\nTuning Results (sorted by ISO PR-AUC):")
    print(tuning_df.sort_values('ISO_PR-AUC', ascending=False).to_string(index=False))
    
    best_iso = tuning_df.loc[tuning_df['ISO_PR-AUC'].idxmax(), 'contamination']
    best_lof = tuning_df.loc[tuning_df['LOF_PR-AUC'].idxmax(), 'contamination']
    best_svm = tuning_df.loc[tuning_df['SVM_PR-AUC'].idxmax(), 'contamination']
    
    print(f"\nBest contamination:")
    print(f"  Isolation Forest : {best_iso:.5f}")
    print(f"  LOF              : {best_lof:.5f}")
    print(f"  One-Class SVM    : {best_svm:.5f}")
    
    return best_iso, best_lof, best_svm