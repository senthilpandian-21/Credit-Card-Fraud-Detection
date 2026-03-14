import matplotlib.pyplot as plt
import seaborn as sns

def plot_eda(y):
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title('Class Distribution (highly imbalanced)')
    plt.xlabel('Class (0 = Normal, 1 = Fraud)')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    plt.close()
    
    print("EDA plots saved: class_distribution.png")