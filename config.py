"""Pipeline configuration constants."""

RANDOM_SEED = 42
DATA_PATH = "data/dataset.csv"
TEST_SIZE = 0.2
N_FOLDS = 3

# Full hyperparameter space considered for the model.
# The grid search in pipeline.py samples a reduced subset of this for speed.
XGBOOST_PARAMS = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "min_samples_split": [2, 5, 10],
    "subsample": [0.8, 1.0],
}

MIN_ACCURACY = 0.88
MIN_DROPOUT_RECALL = 0.80

TARGET_CLASSES = ["Dropout", "Enrolled", "Graduate"]

SOCIOECONOMIC_FEATURES = [
    "Marital status", "Application mode", "Application order",
    "Daytime/evening attendance", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Displaced",
    "Debtor", "Tuition fees up to date", "Gender",
    "Scholarship holder", "International"
]

ACADEMIC_FEATURES = [
    "Previous qualification (grade)", "Admission grade",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Age at enrollment", "Course"
]

MACROECONOMIC_FEATURES = [
    "Unemployment rate", "Inflation rate", "GDP"
]
