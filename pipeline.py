import os
import kfp
from kfp import dsl
from kfp.dsl import Output, Dataset, Model, Metrics, Input
from google.cloud import aiplatform
from typing import NamedTuple

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "temporal-nebula-492415-h5")
REGION = "us-central1"
BUCKET_NAME = "gs://fraud-detection-artifacts"
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root"

# --- TASK 7: DRIFT SIMULATION COMPONENT ---
@dsl.component(base_image='python:3.9', packages_to_install=['pandas', 'numpy'])
def ingestion_with_drift(dataset: Output[Dataset], drifted_dataset: Output[Dataset]):
    import pandas as pd
    import numpy as np
    
    # Base Data (Task 1-4)
    data = {
        'isFraud': [0, 1, 0, 0, 1, 0] * 100,
        'TransactionAmt': [100.0, 20.0, 50.0, 300.0, 15.0, 45.0] * 100,
        'DeviceInfo': ['Windows', 'iOS', 'Windows', 'Android', 'iOS', 'Windows'] * 100,
        'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', 'gmail.com', 'outlook.com', 'gmail.com'] * 100,
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset.path, index=False)
    
    # Simulate Time-Based Drift (Task 7)
    # Pattern change: Transaction amounts increase by 5x (Shock drift)
    df_drifted = df.copy()
    df_drifted['TransactionAmt'] = df_drifted['TransactionAmt'] * 5
    df_drifted.to_csv(drifted_dataset.path, index=False)

@dsl.component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
def preprocessing(input_df: Input[Dataset], output_df: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(input_df.path)
    # Task 8/Privacy: Basic masking simulation
    df.fillna(method='ffill', inplace=True)
    df.to_csv(output_df.path, index=False)

@dsl.component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
def feature_eng(input_df: Input[Dataset], output_df: Output[Dataset]):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(input_df.path)
    for col in ['DeviceInfo', 'P_emaildomain']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df.to_csv(output_df.path, index=False)

# --- TASK 9: EXPLAINABILITY & COST-SENSITIVE TRAINING ---
@dsl.component(
    base_image='python:3.9', 
    packages_to_install=['pandas', 'scikit-learn', 'xgboost', 'numpy']
)
def cost_sensitive_training_with_xai(
    input_df: Input[Dataset], 
    metrics: Output[Metrics]
) -> NamedTuple('Outputs', [('weighted_auc', float), ('is_drift_detected', bool)]):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from xgboost import XGBClassifier
    from collections import namedtuple

    df = pd.read_csv(input_df.path)
    X = df.drop('isFraud', axis=1)
    y = df['isFraud'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cost-Sensitive Training
    model = XGBClassifier(n_estimators=50, scale_pos_weight=5) 
    model.fit(X_train, y_train)
    
    # Task 9: Feature Importance (Explainability)
    # Iska output logs mein aayega for proof
    importance = model.feature_importances_
    for i, val in enumerate(importance):
        print(f"EXPLAINABILITY_LOG: Feature {X.columns[i]} Importance Score: {val:.4f}")

    preds = model.predict(X_test)
    auc_val = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Task 7: Simple Drift Logic for demo
    # If mean transaction amount in test set is > 200, drift is True
    is_drift = bool(X_test['TransactionAmt'].mean() > 200)

    metrics.log_metric("Weighted_AUC", auc_val)
    metrics.log_metric("Drift_Detected", 1 if is_drift else 0)
    
    outputs = namedtuple('Outputs', ['weighted_auc', 'is_drift_detected'])
    return outputs(auc_val, is_drift)

@dsl.pipeline(name="fraud-detection-final-pipeline")
def final_pipeline():
    ingest = ingestion_with_drift()
    
    # Normal Path
    pre = preprocessing(input_df=ingest.outputs['dataset']).after(ingest)
    feat = feature_eng(input_df=pre.outputs['output_df']).after(pre)
    train = cost_sensitive_training_with_xai(input_df=feat.outputs['output_df']).after(feat)

    # Task 8: Retraining Logic (Condition)
    with dsl.If(train.outputs['is_drift_detected'] == True):
        # Triggering retraining on drifted data
        pre_drift = preprocessing(input_df=ingest.outputs['drifted_dataset'])
        feat_drift = feature_eng(input_df=pre_drift.outputs['output_df'])
        train_drift = cost_sensitive_training_with_xai(input_df=feat_drift.outputs['output_df'])

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline_func=final_pipeline, package_path='final_pipeline.yaml')
    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.PipelineJob(
        display_name="final-task-7-8-9-run",
        template_path="final_pipeline.yaml",
        pipeline_root=PIPELINE_ROOT
    )
    job.run()
