from data_loader import load_and_preprocess_data
from eda import plot_eda
from Tuning import run_tuning
from model import train_and_evaluate
from Visualize import plot_anomalies

def main():
    X, y = load_and_preprocess_data(r"D:\\NEW_VOLUME_E\\ML_PROJECTS\\UNSUPERVISED_PROJECTS\\FRAUD_DETECTION\\data\\creditcard.csv")
    plot_eda(y)
    best_iso, best_lof, best_svm = run_tuning(X, y)
    results = train_and_evaluate(
        X, y,
        best_iso, best_lof, best_svm,
        save_dir="saved_models"
    )
   
    plot_anomalies(X, results)
    
    print("\nProject complete!")
    print("Models are saved in the 'saved_models' folder.")
    print("You can now load them later with: model = joblib.load('path/to/model.joblib')")

if __name__ == "__main__":
    main()