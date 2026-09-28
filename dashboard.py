"""Desktop dashboard for the student dropout prediction pipeline."""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH
from data_loader import DataLoader
from pipeline import Pipeline
from predict import load_artifacts, MODEL_PATH

LIME_GREEN = "#32CD32"
LIME_GREEN_DARK = "#279627"
WHITE = "#FFFFFF"
TEXT_DARK = "#1A1A1A"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Dashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Student Dropout Risk Dashboard")
        self.geometry("950x680")
        self.configure(bg=WHITE)

        self._build_style()
        self._build_header()
        self._build_actions()
        self._build_notebook()

    # ------------------------------------------------------------------ style
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
                        foreground=TEXT_DARK, rowheight=24)
        style.configure("Treeview.Heading", background=LIME_GREEN, foreground=WHITE,
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview.Heading", background=[("active", LIME_GREEN_DARK)])
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(12, 6))

    # ----------------------------------------------------------------- header
    def _build_header(self):
        header = tk.Frame(self, bg=LIME_GREEN, height=60)
        header.pack(fill="x", side="top")
        tk.Label(header, text="Student Dropout Risk Dashboard", bg=LIME_GREEN, fg=WHITE,
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=20, pady=12)
        tk.Label(header, text="XGBoost", bg=LIME_GREEN, fg=WHITE,
                 font=("Segoe UI", 9)).pack(side="right", padx=20)

    # ---------------------------------------------------------------- actions
    def _build_actions(self):
        bar = tk.Frame(self, bg=WHITE, pady=8)
        bar.pack(fill="x", padx=20)
        ttk.Button(bar, text="Train Model", style="Lime.TButton",
                   command=self.on_train).pack(side="left", padx=(0, 10))
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=150)
        self.progress.pack(side="right", padx=(10, 0))
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self.status_var, bg=WHITE, fg=LIME_GREEN_DARK,
                 font=("Segoe UI", 10, "bold")).pack(side="right")

    # --------------------------------------------------------------- notebook
    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._tab_home = tk.Frame(self.nb, bg=WHITE)
        self._tab_perf = tk.Frame(self.nb, bg=WHITE)
        self._tab_early = tk.Frame(self.nb, bg=WHITE)
        self._tab_interventions = tk.Frame(self.nb, bg=WHITE)

        self.nb.add(self._tab_home, text="  Predictions  ")
        self.nb.add(self._tab_perf, text="  Model Performance  ")
        self.nb.add(self._tab_early, text="  Early Warning  ")
        self.nb.add(self._tab_interventions, text="  Interventions  ")
        self._build_predictions_tab()
        self._build_performance_tab()
        self._build_early_warning_tab()
        self._build_interventions_tab()

    # ---------------------------------------------------- tab: predictions
    def _build_predictions_tab(self):
        toolbar = tk.Frame(self._tab_home, bg=WHITE)
        toolbar.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(toolbar, text="Filter:", bg=WHITE, fg=TEXT_DARK,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 5))
        self._filter_var = tk.StringVar(value="All")
        ttk.Combobox(toolbar, textvariable=self._filter_var, state="readonly", width=12,
                     values=["All", "Dropout", "Enrolled", "Graduate"]).pack(side="left")
        self._filter_var.trace_add("write", lambda *_: self._apply_filter())
        ttk.Button(toolbar, text="Refresh", style="Lime.TButton",
                   command=self._refresh).pack(side="left", padx=(10, 0))

        cols = ("student", "predicted_outcome", "dropout_probability", "at_risk", "risk_tier")
        self.tree = ttk.Treeview(self._tab_home, columns=cols, show="headings")
        for col, lbl in zip(cols, ("Student ID", "Predicted Outcome", "Dropout Probability", "At Risk", "Risk Tier")):
            self.tree.heading(col, text=lbl)
            self.tree.column(col, anchor="center")
        sb = ttk.Scrollbar(self._tab_home, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        sb.pack(side="right", fill="y", pady=10)

    # ------------------------------------------------- tab: model performance
    def _build_performance_tab(self):
        # metrics row
        metrics_frame = tk.LabelFrame(self._tab_perf, text="Metrics", bg=WHITE,
                                      fg=LIME_GREEN_DARK, font=("Segoe UI", 10, "bold"), bd=2)
        metrics_frame.pack(fill="x", padx=15, pady=(15, 8))
        self._metric_vars = {}
        for col, label in enumerate(("Accuracy", "AUC-ROC", "Dropout Recall", "Dropout F1")):
            tk.Label(metrics_frame, text=label, bg=WHITE, fg=TEXT_DARK,
                     font=("Segoe UI", 9, "bold")).grid(row=0, column=col, padx=25, pady=(6, 2))
            var = tk.StringVar(value="—")
            self._metric_vars[label] = var
            tk.Label(metrics_frame, textvariable=var, bg=WHITE, fg=LIME_GREEN_DARK,
                     font=("Segoe UI", 14, "bold")).grid(row=1, column=col, padx=25, pady=(0, 8))

        # confusion matrix
        self._cm_frame = tk.LabelFrame(self._tab_perf, text="Confusion Matrix", bg=WHITE,
                                       fg=LIME_GREEN_DARK, font=("Segoe UI", 10, "bold"), bd=2)
        self._cm_frame.pack(fill="x", padx=15, pady=(0, 10))

    # ------------------------------------------------- tab: early warning
    _FLAGS = [
        ("Debtor",                                lambda v: float(v) == 1,   "Has outstanding debt"),
        ("Tuition fees up to date",               lambda v: float(v) == 0,   "Tuition fees not paid"),
        ("Scholarship holder",                    lambda v: float(v) == 0,   "No scholarship"),
        ("Curricular units 1st sem (approved)",   lambda v: float(v) < 3,    "Passed fewer than 3 units in semester 1"),
        ("Curricular units 1st sem (grade)",      lambda v: float(v) < 10,   "Average grade below 10 in semester 1"),
        ("Curricular units 2nd sem (approved)",   lambda v: float(v) < 3,    "Passed fewer than 3 units in semester 2"),
        ("Curricular units 2nd sem (grade)",      lambda v: float(v) < 10,   "Average grade below 10 in semester 2"),
        ("Curricular units 1st sem (enrolled)",   lambda v: float(v) == 0,   "Not enrolled in any units in semester 1"),
        ("Curricular units 2nd sem (enrolled)",   lambda v: float(v) == 0,   "Not enrolled in any units in semester 2"),
        ("Age at enrollment",                     lambda v: float(v) > 30,   "Enrolled at age over 30"),
    ]

    def _build_early_warning_tab(self):
        self._ew_student_features = {}
        tk.Label(self._tab_early, text="Early Warning — At-Risk Students", bg=WHITE,
                 fg=LIME_GREEN_DARK, font=("Segoe UI", 13, "bold")).pack(pady=(16, 2))
        tk.Label(self._tab_early, text="Double-click a student to view full details and add an intervention.",
                 bg=WHITE, fg=TEXT_DARK, font=("Segoe UI", 9, "italic")).pack(pady=(0, 6))
        ttk.Button(self._tab_early, text="Load At-Risk Students", style="Lime.TButton",
                   command=self._load_early_warning).pack(pady=(0, 8))
        cols = ("student_id", "dropout_prob", "warning_flags")
        self._ew_tree = ttk.Treeview(self._tab_early, columns=cols, show="headings")
        for col, lbl, w in zip(cols, ("Student ID", "Dropout Prob", "Warning Flags"), (100, 100, 600)):
            self._ew_tree.heading(col, text=lbl)
            self._ew_tree.column(col, anchor="w", width=w)
        sb = ttk.Scrollbar(self._tab_early, orient="vertical", command=self._ew_tree.yview)
        self._ew_tree.configure(yscrollcommand=sb.set)
        self._ew_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=4)
        sb.pack(side="right", fill="y", pady=4)
        self._ew_tree.bind("<Double-1>", self._on_intervene)

    def _load_early_warning(self):
        import csv
        predictions_csv = os.path.join(BASE_DIR, "output", "predictions.csv")
        dataset_csv     = os.path.join(BASE_DIR, DATA_PATH)
        if not os.path.exists(predictions_csv):
            messagebox.showerror("No predictions", "Train the model first."); return
        self._ew_student_features = {}
        with open(dataset_csv, encoding="utf-8-sig") as f:
            first = f.readline(); f.seek(0)
            delim = ";" if ";" in first and "," not in first else ","
            for i, row in enumerate(csv.DictReader(f, delimiter=delim)):
                self._ew_student_features[f"S{i:05d}"] = {
                    k: v for k, v in row.items() if k != "Target" and v not in ("", None)}
        self._ew_tree.delete(*self._ew_tree.get_children())
        with open(predictions_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["at_risk"].lower() != "true": continue
                feats = self._ew_student_features.get(row["student_id"], {})
                flags = [label for col, check, label in self._FLAGS
                         if col in feats and check(feats[col])]
                self._ew_tree.insert("", "end", values=(
                    row["student_id"], row["dropout_probability"],
                    "  |  ".join(flags) if flags else "No specific flags"))

    def _on_intervene(self, _event):
        item = self._ew_tree.focus()
        if not item: return
        sid, prob, flags = self._ew_tree.item(item, "values")
        popup = tk.Toplevel(self, bg=WHITE)
        popup.title(f"Intervene — {sid}")
        popup.geometry("480x380")
        popup.grab_set()
        tk.Label(popup, text=f"Student: {sid}  |  Dropout Prob: {prob}",
                 bg=WHITE, fg=LIME_GREEN_DARK, font=("Segoe UI", 10, "bold")).pack(pady=(14, 4))
        flag_frame = tk.LabelFrame(popup, text="Warning Flags", bg=WHITE,
                                   fg=LIME_GREEN_DARK, font=("Segoe UI", 9, "bold"))
        flag_frame.pack(fill="x", padx=16, pady=(0, 12))
        for flag in flags.split("  |  "):
            tk.Label(flag_frame, text=f"⚠ {flag}", bg=WHITE, fg=TEXT_DARK,
                     font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=8, pady=2)
        tk.Label(popup, text="Intervention Note:", bg=WHITE, fg=TEXT_DARK,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=16)
        note = tk.Text(popup, height=5, font=("Segoe UI", 10), wrap="word",
                       highlightbackground=LIME_GREEN, highlightthickness=1)
        note.pack(fill="x", padx=16, pady=(4, 12))
        def save():
            import csv; from datetime import datetime
            text = note.get("1.0", "end-1c").strip()
            if not text:
                messagebox.showwarning("Empty note", "Please enter an intervention note before saving.", parent=popup)
                return
            path = os.path.join(BASE_DIR, "output", "interventions.csv")
            write_header = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header: w.writerow(["student_id", "dropout_probability", "warning_flags", "intervention_note", "timestamp"])
                w.writerow([sid, prob, flags, text, datetime.now().strftime("%Y-%m-%d %H:%M")])
            self._ew_tree.item(item, values=(f"✓ {sid}", prob, flags))
            self._load_interventions()
            popup.destroy()
        ttk.Button(popup, text="Save Intervention", style="Lime.TButton", command=save).pack(pady=(0, 10))

    # ------------------------------------------------- tab: interventions
    def _build_interventions_tab(self):
        tk.Label(self._tab_interventions, text="Intervention History", bg=WHITE,
                 fg=LIME_GREEN_DARK, font=("Segoe UI", 13, "bold")).pack(pady=(16, 2))
        ttk.Button(self._tab_interventions, text="Refresh", style="Lime.TButton",
                   command=self._load_interventions).pack(pady=(0, 8))
        cols = ("student_id", "dropout_probability", "warning_flags", "intervention_note", "timestamp")
        self._int_tree = ttk.Treeview(self._tab_interventions, columns=cols, show="headings")
        for col, lbl, w in zip(cols, ("Student ID", "Dropout Prob", "Warning Flags", "Note", "Timestamp"),
                               (90, 90, 220, 260, 130)):
            self._int_tree.heading(col, text=lbl)
            self._int_tree.column(col, anchor="w", width=w)
        sb = ttk.Scrollbar(self._tab_interventions, orient="vertical", command=self._int_tree.yview)
        self._int_tree.configure(yscrollcommand=sb.set)
        self._int_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=4)
        sb.pack(side="right", fill="y", pady=4)

    def _load_interventions(self):
        import csv
        path = os.path.join(BASE_DIR, "output", "interventions.csv")
        self._int_tree.delete(*self._int_tree.get_children())
        if not os.path.exists(path): return
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._int_tree.insert("", "end", values=(
                    row.get("student_id", ""), row.get("dropout_probability", ""),
                    row.get("warning_flags", "—"), row.get("intervention_note", ""), row.get("timestamp", "")))

    # ----------------------------------------------------------------- train
    def on_train(self):
        data_path = os.path.join(BASE_DIR, DATA_PATH)
        if not os.path.exists(data_path):
            messagebox.showerror("Dataset not found", f"No dataset at:\n{data_path}")
            return
        self.status_var.set("Training...")
        self.progress.start(10)
        threading.Thread(target=self._run_training, args=(data_path,), daemon=True).start()

    def _run_training(self, data_path):
        try:
            loader = DataLoader(data_path)
            students = loader.get_students(loader.validate(loader.load()))

            output_dir = os.path.join(BASE_DIR, "output")
            pipeline = Pipeline(students, output_dir=output_dir)
            pipeline.run()

            predictions_csv = os.path.join(output_dir, "predictions.csv")
            if os.path.exists(predictions_csv):
                self.after(200, self._populate_results_from_csv,
                           self._read_predictions_csv(predictions_csv))

            self.after(200, self._populate_metrics, pipeline.metrics)
            self.after(200, self.status_var.set, "Training complete.")
            self.after(200, self.progress.stop)
        except Exception as exc:
            self.after(0, messagebox.showerror, "Training failed", str(exc))
            self.after(0, self.status_var.set, "Training failed.")
            self.after(0, self.progress.stop)

    # ------------------------------------------------------------ populators
    def _read_predictions_csv(self, path):
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _risk_tier(self, prob):
        p = float(prob)
        return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"

    def _populate_results_from_csv(self, results):
        self.tree.delete(*self.tree.get_children())
        for r in results:
            prob = r.get("dropout_probability", 0)
            self.tree.insert("", "end", values=(
                r.get("student_id", ""),
                r.get("predicted_outcome", ""),
                f"{float(prob):.4f}",
                "Yes" if r.get("at_risk", "").lower() == "true" else "No",
                self._risk_tier(prob),
            ))

    def _populate_metrics(self, metrics):
        m = metrics
        self._metric_vars["Accuracy"].set(f"{m.accuracy:.4f}")
        self._metric_vars["AUC-ROC"].set(f"{m.auc_roc:.4f}")
        self._metric_vars["Dropout Recall"].set(f"{m.get_dropout_recall():.4f}")
        self._metric_vars["Dropout F1"].set(f"{m.f1_per_class.get('Dropout', 0):.4f}")

        for w in self._cm_frame.winfo_children():
            w.destroy()
        classes = m.classes
        tk.Label(self._cm_frame, text="Actual \\ Pred", bg=WHITE, fg=TEXT_DARK,
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
                         font=("Segoe UI", 10), width=10, relief="groove").grid(
                             row=i+1, column=j+1, padx=2, pady=2)

    def _apply_filter(self):
        selected = self._filter_var.get()
        predictions_csv = os.path.join(BASE_DIR, "output", "predictions.csv")
        if not os.path.exists(predictions_csv):
            return
        all_rows = self._read_predictions_csv(predictions_csv)
        filtered = [r for r in all_rows if selected == "All" or r.get("predicted_outcome") == selected]
        self._populate_results_from_csv(filtered)

    def _refresh(self):
        predictions_csv = os.path.join(BASE_DIR, "output", "predictions.csv")
        if os.path.exists(predictions_csv):
            self._populate_results_from_csv(self._read_predictions_csv(predictions_csv))
        self.status_var.set("Refreshed.")


if __name__ == "__main__":
    Dashboard().mainloop()
