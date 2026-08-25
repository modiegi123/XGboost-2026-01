"""Main entry point for the dropout prediction system."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH, RANDOM_SEED
from data_loader import DataLoader
from pipeline import Pipeline


def main():
    print("\n" + "=" * 70)
    print("  Enhancing Transparency and Early Detection of At-Risk Students")
    print("  in Higher Education Using Academic and Socioeconomic Indicators")
    print("  " + "-" * 60)
    print("  Author: Josephine Morudu (218075774)")
    print("  University of Johannesburg")
    print("=" * 70)

    print("\n[DATA INGESTION] Loading dataset...")
    data_path = os.path.join(os.path.dirname(__file__), DATA_PATH)

    if not os.path.exists(data_path):
        print(f"\n  ERROR: Dataset not found at: {data_path}")
        print(f"  Please download the 'Predict Students Dropout and Academic Success'")
        print(f"  dataset from UCI ML Repository or Kaggle and place it at:")
        print(f"  {data_path}")
        print(f"\n  The file should be named 'dataset.csv' in the 'data/' folder.")
        sys.exit(1)

    loader = DataLoader(data_path)
    records = loader.load()
    records = loader.validate(records)
    students = loader.get_students(records)

    print(loader.get_load_report())
    print(f"  Student objects created: {len(students)}")

    class_dist = {}
    for s in students:
        class_dist[s.outcome] = class_dist.get(s.outcome, 0) + 1
    print(f"  Class distribution: {class_dist}")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    pipeline = Pipeline(students, output_dir=output_dir)
    pipeline.run()

    print("\n" + "=" * 70)
    print("  OUTPUT ARTEFACTS:")
    print("  " + "-" * 60)
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        size = os.path.getsize(filepath)
        print(f"  {filename:<40} ({size:>8} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
