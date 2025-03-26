import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Proctoring Analysis")
        self.root.geometry("800x600")

        # Main frame
        self.main_frame = Frame(self.root)
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Button to generate curves
        self.analyze_button = Button(self.main_frame, text="Generate Analysis", command=self.generate_analysis)
        self.analyze_button.pack(pady=20)

    def generate_analysis(self):
        # Load logged data
        try:
            data = pd.read_csv("malicious_activity_log.txt", names=["timestamp", "activity"], sep=": ")
            data["is_true_positive"] = data["activity"].apply(lambda x: 1 if "detected" in x.lower() else 0)
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        # Extract true labels and predicted scores (example scoring)
        y_true = data["is_true_positive"]
        y_scores = data["activity"].apply(lambda x: 1 if "detected" in x.lower() else 0)  # Example scoring

        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Display Precision-Recall curve
        self.display_curve(recall, precision, "Precision-Recall Curve", "Recall", "Precision", f"AUC = {pr_auc:.2f}")

        # Display ROC curve
        self.display_curve(fpr, tpr, "ROC Curve", "False Positive Rate", "True Positive Rate", f"AUC = {roc_auc:.2f}")

    def display_curve(self, x, y, title, xlabel, ylabel, auc_label):
        fig, ax = plt.subplots()
        ax.plot(x, y, label=auc_label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

def main():
    root = Tk()
    app = AnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()