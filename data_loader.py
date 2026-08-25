"""Dataset loading and validation."""

import csv
import os


class DataLoader:

    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_records = []
        self.feature_names = []
        self.target_column = "Target"
        self.n_records_loaded = 0
        self.n_duplicates_removed = 0
        self.alerts = []

    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        records = []
        with open(self.data_path, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            f.seek(0)

            # UCI format uses semicolons
            if ';' in first_line and ',' not in first_line:
                delimiter = ';'
            else:
                delimiter = ','

            reader = csv.DictReader(f, delimiter=delimiter)
            self.feature_names = [col for col in reader.fieldnames
                                  if col != self.target_column]

            for row in reader:
                processed = {}
                for key, value in row.items():
                    try:
                        processed[key] = float(value)
                    except (ValueError, TypeError):
                        processed[key] = value
                records.append(processed)

        self.n_records_loaded = len(records)
        self.raw_records = records
        return records

    def validate(self, records=None):
        if records is None:
            records = self.raw_records

        seen_hashes = set()
        unique_records = []
        for i, record in enumerate(records):
            record_key = tuple(sorted(record.items()))
            if record_key in seen_hashes:
                self.n_duplicates_removed += 1
                self.alerts.append(f"Duplicate record found at index {i}")
            else:
                seen_hashes.add(record_key)
                unique_records.append(record)

        if self.n_duplicates_removed > 0:
            self.alerts.append(
                f"ALERT: {self.n_duplicates_removed} duplicate records removed.")

        expected = self.n_records_loaded - self.n_duplicates_removed
        actual = len(unique_records)
        if actual != expected:
            self.alerts.append(
                f"WARNING: Expected {expected} records, got {actual}. Possible data loss!")

        return unique_records

    def get_students(self, records=None):
        from models.student import Student

        if records is None:
            records = self.raw_records

        students = []
        for i, record in enumerate(records):
            outcome = record.get(self.target_column, "Unknown")
            if isinstance(outcome, float):
                outcome = str(int(outcome))

            outcome_map = {"Dropout": "Dropout", "Graduate": "Graduate",
                          "Enrolled": "Enrolled",
                          "0": "Dropout", "1": "Enrolled", "2": "Graduate"}
            outcome = outcome_map.get(outcome, outcome)

            features = {k: v for k, v in record.items() if k != self.target_column}

            student = Student(
                student_id=f"S{i:05d}",
                features=features,
                outcome=outcome
            )
            students.append(student)

        return students

    def get_load_report(self):
        report = f"Data Loading Report:\n"
        report += f"  Source: {self.data_path}\n"
        report += f"  Records loaded: {self.n_records_loaded}\n"
        report += f"  Duplicates removed: {self.n_duplicates_removed}\n"
        report += f"  Features: {len(self.feature_names)}\n"
        if self.alerts:
            report += f"  Alerts:\n"
            for alert in self.alerts:
                report += f"    - {alert}\n"
        return report
