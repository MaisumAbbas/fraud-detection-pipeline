# Task 4

from kfp import dsl
from kfp.dsl import Output, Dataset, Model, Metrics, Input
from google.cloud import aiplatform
from typing import NamedTuple

PROJECT_ID = "temporal-nebula-492415-h5"
REGION = "us-central1"
BUCKET_NAME = "gs://fraud-detection-artifacts"
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root"

# --- COMPONENTS (Pre-requisites from Tasks 1-3) ---

@dsl.component(base_image='python:3.9', packages_to_install=['pandas'])
def ingestion(dataset: Output[Dataset]):
    import pandas as pd
    data = {
        'isFraud': [0, 1, 0, 0, 1, 0] * 100,
        'TransactionAmt': [100.5, 20.0, 50.0, 300.0, 15.0, 45.0] * 100,
        'DeviceInfo': ['Windows', 'iOS', 'Windows', 'Android', 'iOS', 'Windows'] * 100,
        'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', 'gmail.com', 'outlook.com', 'gmail.com'] * 100,
    }
    pd.DataFrame(data).to_csv(dataset.path, index=False)

@dsl.component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
def preprocessing(input_df: Input[Dataset], output_df: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(input_df.path)
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

# --- TASK 4 COMPONENT: COST-SENSITIVE TRAINING ---

@dsl.component(
    base_image='python:3.9', 
    packages_to_install=['pandas', 'scikit-learn', 'xgboost']
)
def cost_sensitive_training(
    input_df: Input[Dataset], 
    metrics: Output[Metrics]
) -> NamedTuple('Outputs', [('weighted_auc', float), ('cost_savings', float)]):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from xgboost import XGBClassifier
    from collections import namedtuple

    # 1. Load Data
    df = pd.read_csv(input_df.path)
    X = df.drop('isFraud', axis=1)
    y = df['isFraud'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Standard Training (No Penalty) ---
    model_std = XGBClassifier(n_estimators=50)
    model_std.fit(X_train, y_train)
    preds_std = model_std.predict(X_test)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_test, preds_std).ravel()

    # --- Cost-Sensitive Training (High Penalty for False Negatives) ---
    # Assignment Requirement: Assign higher penalty to false negatives
    # 'scale_pos_weight' 1 se zyada rakhne ka matlab hai Fraud (1) ko zyada importance dena
    model_cost = XGBClassifier(n_estimators=50, scale_pos_weight=5) 
    model_cost.fit(X_train, y_train)
    preds_cost = model_cost.predict(X_test)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test, preds_cost).ravel()

    # --- Business Impact Analysis ---
    # فرض کریں: False Negative (Fraud miss hona) = $500 nuksan
    # False Positive (Ghalat alert) = $20 nuksan
    cost_std = (fn_s * 500) + (fp_s * 20)
    cost_cost_sensitive = (fn_c * 500) + (fp_c * 20)
    savings = float(cost_std - cost_cost_sensitive)
    
    auc_val = float(roc_auc_score(y_test, model_cost.predict_proba(X_test)[:, 1]))

    # Dashboard Metrics
    metrics.log_metric("Standard_Loss_USD", float(cost_std))
    metrics.log_metric("CostSensitive_Loss_USD", float(cost_cost_sensitive))
    metrics.log_metric("Business_Savings_USD", savings)
    metrics.log_metric("Weighted_AUC", auc_val)

    print(f"Standard Model Cost: ${cost_std}")
    print(f"Cost-Sensitive Model Cost: ${cost_cost_sensitive}")
    print(f"Confusion Matrix (Cost-Sensitive):\n{confusion_matrix(y_test, preds_cost)}")
    
    outputs = namedtuple('Outputs', ['weighted_auc', 'cost_savings'])
    return outputs(auc_val, savings)

@dsl.component(base_image='python:3.9')
def deployment_step():
    print("Deployment triggered: Model is cost-optimized.")

# --- PIPELINE DEFINITION ---

@dsl.pipeline(name="fraud-detection-task4-full")
def pipeline_task4(threshold: float = 0.5):
    ingest = ingestion()
    pre = preprocessing(input_df=ingest.outputs['dataset']).after(ingest)
    feat = feature_eng(input_df=pre.outputs['output_df']).after(pre)
    
    # Task 4 Execution
    cost_eval = cost_sensitive_training(input_df=feat.outputs['output_df']).after(feat)
    
    # Conditional logic based on AUC
    with dsl.If(cost_eval.outputs['weighted_auc'] > threshold):
        deployment_step().after(cost_eval)

# --- COMPILE & RUN ---
kfp.compiler.Compiler().compile(pipeline_func=pipeline_task4, package_path='task4.yaml')

aiplatform.init(project=PROJECT_ID, location=REGION)
job = aiplatform.PipelineJob(
    display_name="task-4-cost-sensitive-run",
    template_path="task4.yaml",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={'threshold': 0.5}
)
job.run()
