

class Student:
    """Represents a student with features and outcome."""

    def __init__(self, student_id, features, outcome=None):
        self.student_id = student_id
        self.features = features
        self.outcome = outcome

    def get_academic_profile(self):
        from config import ACADEMIC_FEATURES
        return {k: v for k, v in self.features.items() if k in ACADEMIC_FEATURES}

    def get_socioeconomic_profile(self):
        from config import SOCIOECONOMIC_FEATURES
        return {k: v for k, v in self.features.items() if k in SOCIOECONOMIC_FEATURES}

    def get_macroeconomic_profile(self):
        from config import MACROECONOMIC_FEATURES
        return {k: v for k, v in self.features.items() if k in MACROECONOMIC_FEATURES}

    def to_feature_vector(self, feature_names):
        return [self.features.get(name, 0.0) for name in feature_names]

    def __repr__(self):
        return f"Student(id={self.student_id}, outcome={self.outcome})"
