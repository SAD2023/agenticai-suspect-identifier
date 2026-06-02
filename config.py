"""
set dataset params here e.g. columns, etc
"""

import os

# ---------------------------------------------------------------------------
# RF MODEL CONFIGS

# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "n_estimators":    200,        # Number of decision trees in the forest.
    "max_depth":       12,         # Max tree depth; limits overfitting.
    "min_samples_split": 10,       # Min samples needed to split a node.
    "min_samples_leaf":  5,        # Min samples required at a leaf node.
    "class_weight":    "balanced", # Compensates for imbalanced classes (few suspects vs. many non-suspects).
    "random_state":    42,         
    "n_jobs":          -1,         
}

# 
# FEATURE CONFIGURATION
# 
FEATURE_CONFIG = {

    # column names from inputs
    "FEATURE_COLUMNS": ["watchlist_match_score",
    "prior_security_incidents",
    "trips_to_risk_zones",
    "last_minute_booking_rate",
    "document_anomaly_score",
    "network_association_score",
    "financial_activity_flag",
    "cross_agency_flag_count",
    "border_screening_anomaly_score",
    "encrypted_comms_flag",],  


    # suspect yes or not confirmation column
    "TARGET_COLUMN": "confirmed_case", 

    # non numeric columns
    "CATEGORICAL_COLUMNS": []  # TODO: Define after ethics committee dataset review.

}


# 
# Analysts see each scored individual classified into one of three tiers.
# 

PRIORITY_CONFIG = {
    "HIGH":   0.70,  # risk_score >= 0.70  →  HIGH   (recommend immediate analyst review)
    "MEDIUM": 0.40,  # risk_score >= 0.40  →  MEDIUM (review when capacity allows)
                     # risk_score <  0.40  →  LOW    (de-prioritize)
}


FILE_CONFIG = {
    "MODEL_SAVE_PATH":        "models/trained_model.joblib",
    "MODEL_METADATA_PATH":    "models/model_metadata.joblib",
    "DEFAULT_OUTPUT_DIR":     "outputs/",
    "SUPPORTED_INPUT_FORMATS": [".xlsx", ".xls", ".csv", ".pdf"],
}

for _dir in ("models", "outputs", "logs"):
    os.makedirs(_dir, exist_ok=True)
