"""Desktop dashboard for the student dropout prediction pipeline.

Built with tkinter (Python standard library only, matching the rest of the
project's no-third-party-dependency constraint). Lets a user train the
model on data/dataset.csv or predict on a new CSV, and shows the results
without touching the terminal.
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH
from data_loader import DataLoader
from pipeline import Pipeline
from predict import load_artifacts, read_students_csv, predict_students, MODEL_PATH

LIME_GREEN = "#32CD32"
LIME_GREEN_DARK = "#279627"
WHITE = "#FFFFFF"
TEXT_DARK = "#1A1A1A"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class _TextRedirector:
    """Sends print() output into a tkinter Text widget instead of stdout."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.after(0, self._append, message)

    def _append(self, message):
        self.widget.configure(state="normal")
        self.widget.insert("end", message)
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass


class Dashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Student Dropout Risk Dashboard")
        self.geometry("900x950")
        self.configure(bg=WHITE)

        self._build_style()
        self._build_header()
        self._build_actions()
        self._build_log()
        self._build_metrics_panel()
        self._build_confusion_matrix()
        self._build_results_table()

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Lime.TButton", background=LIME_GREEN, foreground=WHITE,
                         font=("Segoe UI", 10, "bold"), padding=8, borderwidth=0)
        style.map("Lime.TButton", background=[("active", LIME_GREEN_DARK)])

        style.configure("Treeview", background=WHITE, fieldbackground=WHITE,
                         foreground=TEXT_DARK, rowheight=24, bordercolor=LIME_GREEN)
        style.configure("Treeview.Heading", background=LIME_GREEN, foreground=WHITE,
                         font=("Segoe UI", 9, "bold"))
        style.map("Treeview.Heading", background=[("active", LIME_GREEN_DARK)])

    def _build_header(self):
        header = tk.Frame(self, bg=LIME_GREEN, height=70)
        header.pack(fill="x", side="top")

        tk.Label(header, text="Student Dropout Risk Dashboard", bg=LIME_GREEN, fg=WHITE,
                  font=("Segoe UI", 16, "bold")).pack(side="left", padx=20, pady=15)
        tk.Label(header, text="XGBoost",
                  bg=LIME_GREEN, fg=WHITE, font=("Segoe UI", 9)).pack(side="right", padx=20)

    def _build_actions(self):
        bar = tk.Frame(self, bg=WHITE, pady=10)
        bar.pack(fill="x", padx=20)

        ttk.Button(bar, text="Train Model", style="Lime.TButton",
                   command=self.on_train).pack(side="left", padx=(0, 10))
        ttk.Button(bar, text="Predict from CSV...", style="Lime.TButton",
                   command=self.on_predict).pack(side="left")

        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=150)
        self.progress.pack(side="right", padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self.status_var, bg=WHITE, fg=LIME_GREEN_DARK,
                  font=("Segoe UI", 10, "bold")).pack(side="right")

    def _build_log(self):
        frame = tk.LabelFrame(self, text="Pipeline Log", bg=WHITE, fg=LIME_GREEN_DARK,
                               font=("Segoe UI", 10, "bold"), bd=2, labelanchor="nw")
        frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self.log_text = tk.Text(frame, height=14, bg=WHITE, fg=TEXT_DARK,
                                 insertbackground=TEXT_DARK, state="disabled",
                                 font=("Consolas", 9), wrap="word",
                                 highlightbackground=LIME_GREEN, highlightthickness=1)
        scrollbar = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_metrics_panel(self):
        frame = tk.LabelFrame(self, text="Model Performance", bg=WHITE, fg=LIME_GREEN_DARK,
                               font=("Segoe UI", 10, "bold"), bd=2, labelanchor="nw")
        frame.pack(fill="x", padx=20, pady=(0, 10))

        self._metric_vars = {}
        for col, label in enumerate(("Accuracy", "AUC-ROC", "Dropout Recall", "Dropout F1")):
            tk.Label(frame, text=label, bg=WHITE, fg=TEXT_DARK,
                     font=("Segoe UI", 9, "bold")).grid(row=0, column=col, padx=20, pady=(6, 2))
            var = tk.StringVar(value="—")
            self._metric_vars[label] = var
            tk.Label(frame, textvariable=var, bg=WHITE, fg=LIME_GREEN_DARK,
                     font=("Segoe UI", 13, "bold")).grid(row=1, column=col, padx=20, pady=(0, 8))

    def _build_confusion_matrix(self):
        frame = tk.LabelFrame(self, text="Confusion Matrix", bg=WHITE, fg=LIME_GREEN_DARK,
                               font=("Segoe UI", 10, "bold"), bd=2, labelanchor="nw")
        frame.pack(fill="x", padx=20, pady=(0, 10))
        self._cm_frame = frame
        self._cm_labels = {}  # populated after training

    def _populate_metrics(self, metrics):
        m = metrics
        self._metric_vars["Accuracy"].set(f"{m.accuracy:.4f}")
        self._metric_vars["AUC-ROC"].set(f"{m.auc_roc:.4f}")
        self._metric_vars["Dropout Recall"].set(f"{m.get_dropout_recall():.4f}")
        self._metric_vars["Dropout F1"].set(f"{m.f1_per_class.get('Dropout', 0):.4f}")

        # Clear and rebuild confusion matrix grid
        for w in self._cm_frame.winfo_children():
            w.destroy()

        classes = m.classes
        # Header row
        tk.Label(self._cm_frame, text="Actual \ Pred", bg=WHITE, fg=TEXT_DARK,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=8, pady=4)
        for j, cls in enumerate(classes):
            tk.Label(self._cm_frame, text=cls, bg=LIME_GREEN, fg=WHITE,
                     font=("Segoe UI", 9, "bold"), width=10).grid(row=0, column=j+1, padx=2, pady=4)
        for i, actual in enumerate(classes):
            tk.Label(self._cm_frame, text=actual, bg=LIME_GREEN, fg=WHITE,
                     font=("Segoe UI", 9, "bold"), width=12).grid(row=i+1, column=0, padx=2, pady=2)
            for j, pred in enumerate(classes):
                val = m.confusion_matrix.get((actual, pred), 0)
                bg = LIME_GREEN_DARK if actual == pred else WHITE
                fg = WHITE if actual == pred else TEXT_DARK
                tk.Label(self._cm_frame, text=str(val), bg=bg, fg=fg,
                         font=("Segoe UI", 10), width=10, relief="groove").grid(row=i+1, column=j+1, padx=2, pady=2)

    def _build_results_table(self):
        frame = tk.LabelFrame(self, text="Predictions", bg=WHITE, fg=LIME_GREEN_DARK,
                               font=("Segoe UI", 10, "bold"), bd=2, labelanchor="nw")
        frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        columns = ("student", "predicted_outcome", "dropout_probability", "at_risk")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for col, label in zip(columns, ("Student ID", "Predicted Outcome", "Dropout Probability", "At Risk")):
            self.tree.heading(col, text=label)
            self.tree.column(col, anchor="center")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def on_train(self):
        data_path = os.path.join(BASE_DIR, DATA_PATH)
        if not os.path.exists(data_path):
            messagebox.showerror("Dataset not found", f"No dataset at:\n{data_path}")
            return
        self.status_var.set("Training...")
        self.progress.start(10)
        threading.Thread(target=self._run_training, args=(data_path,), daemon=True).start()

    def _run_training(self, data_path):
        old_stdout = sys.stdout
        sys.stdout = _TextRedirector(self.log_text)
        try:
            loader = DataLoader(data_path)
            records = loader.load()
            records = loader.validate(records)
            students = loader.get_students(records)
            print(loader.get_load_report())

            output_dir = os.path.join(BASE_DIR, "output")
            pipeline = Pipeline(students, output_dir=output_dir)
            pipeline.run()

            # Populate table from saved predictions CSV
            predictions_csv = os.path.join(output_dir, "predictions.csv")
            if os.path.exists(predictions_csv):
                results = self._read_predictions_csv(predictions_csv)
                self.after(200, self._populate_results_from_csv, results)

            self.after(200, self._populate_metrics, pipeline.metrics)
            self.after(200, self.status_var.set, "Training complete.")
            self.after(200, self.progress.stop)
        except Exception as exc:
            self.after(0, messagebox.showerror, "Training failed", str(exc))
            self.after(0, self.status_var.set, "Training failed.")
        finally:
            sys.stdout = old_stdout

    def _read_predictions_csv(self, path):
        import csv
        results = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append(row)
        return results

    def _populate_results_from_csv(self, results):
        self.tree.delete(*self.tree.get_children())
        for r in results:
            self.tree.insert("", "end", values=(
                r.get("student_id", ""),
                r.get("predicted_outcome", ""),
                f"{float(r.get('dropout_probability', 0)):.4f}",
                "Yes" if r.get("at_risk", "").lower() == "true" else "No",
            ))
        self.tree.update_idletasks()

    def on_predict(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("No trained model", "Train the model first before predicting.")
            return
        path = filedialog.askopenfilename(title="Select student CSV", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        self.status_var.set("Predicting...")
        threading.Thread(target=self._run_prediction, args=(path,), daemon=True).start()

    def _run_prediction(self, path):
        old_stdout = sys.stdout
        sys.stdout = _TextRedirector(self.log_text)
        try:
            artifacts = load_artifacts()
            feature_rows = read_students_csv(path)
            if not feature_rows:
                raise ValueError("No student rows found in input file.")

            results = predict_students(feature_rows, artifacts)
            print(f"Predicted {len(results)} student row(s) from {os.path.basename(path)}.")
            self.after(0, self._populate_results, results)
            self.after(0, self.status_var.set, "Prediction complete.")
        except Exception as exc:
            self.after(0, messagebox.showerror, "Prediction failed", str(exc))
            self.after(0, self.status_var.set, "Prediction failed.")
        finally:
            sys.stdout = old_stdout

    def _populate_results(self, results):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(results):
            self.tree.insert("", "end", values=(
                f"Student {i + 1}",
                r["predicted_outcome"],
                f"{r['dropout_probability']:.4f}",
                "Yes" if r["at_risk"] else "No",
            ))


if __name__ == "__main__":
    Dashboard().mainloop()
