from open_fe.dataset import Dataset
from open_fe.ops import Column, Operator, OperatorType, ValueType

__all__ = ["FeatureGenerator"]

Feature = Operator


class FeatureGenerator:
    @staticmethod
    def extract_base_features(dataset: Dataset) -> list[Feature]:
        features: list[Feature] = []
        for column_name in dataset.feature_columns:
            feature = Column(column_name)
            feature.value_type = dataset.get_feature_value_type(column_name)
            features.append(feature)
        return features

    @staticmethod
    def enumerate_candidate_features(
        features: list[Feature],
        operators: dict[OperatorType, list[Operator]],
    ) -> list[Operator]:
        # split base features into numeric features and category features.
        category_features = [feature for feature in features if feature.value_type == ValueType.CAT]
        numeric_features = [feature for feature in features if feature.value_type == ValueType.NUM]
        numeric_candidate_features: list[Operator] = []
        category_candidate_features: list[Operator] = []

        for operator in operators[OperatorType.ALL]:
            numeric_candidate_features.extend(operator(feature) for feature in category_features + numeric_features)

        for operator in operators[OperatorType.NUM]:
            numeric_candidate_features.extend(operator(feature) for feature in numeric_features)

        for operator in operators[OperatorType.NUM_NUM]:
            for i, f1 in enumerate(numeric_features):
                numeric_candidate_features.extend(operator(f1, f2) for f2 in numeric_features[i + 1 :])

        for operator in operators[OperatorType.CAT_NUM]:
            for f1 in category_features:
                numeric_candidate_features.extend(operator(f1, f2) for f2 in numeric_features)

        for operator in operators[OperatorType.CAT_CAT]:
            for i, f1 in enumerate(category_features):
                if operator.__class__.__name__ == "Combine":
                    category_candidate_features.extend(operator(f1, f2) for f2 in category_features[i + 1 :])
                else:
                    numeric_candidate_features.extend(operator(f1, f2) for f2 in category_features[i + 1 :])

        return numeric_candidate_features + category_candidate_features
