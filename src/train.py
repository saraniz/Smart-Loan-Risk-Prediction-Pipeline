import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix  # NEW: added for better evaluation
import mlflow
import mlflow.sklearn #mlflow support sklearn models
from sklearn.preprocessing import StandardScaler
from mlflow.data.pandas_dataset import from_pandas

#“Store all experiments inside this SQLite database file”
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Risk_Prediction")

mlflow.set_experiment_tag(
    "mlflow.note.content",
    "This experiment predicts loan risk using Logistic Regression, Random Forest, and XGBoost with preprocessing and scaling."
)
MODEL_PATH = '../models'
DATA_PATH='../data/cleaned_loan.csv'
TARGET = 'target'

def load_data():
    df = pd.read_csv('../data/cleaned_loan.csv')
    print(df.columns)

    return df

def train(df):

    df = df.copy()
    dataset = from_pandas(df, source=DATA_PATH, name="cleaned_loan_v1")

    df = pd.get_dummies(df, columns=['home_ownership','purpose','term'], drop_first=True)
    #seperate features and targets
    X = df.drop(TARGET, axis=1)  #drop the target column. axis=1 mean columns
    y = df[TARGET]

    #train test split
    X_train,X_test,y_train,y_test =train_test_split(X,y, test_size=0.2,random_state=42)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    #define models using python dictionary as key:value pairs
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000), #model try up to 1000 times to find the best fit
        "Random_forest": RandomForestClassifier(n_estimators=100),
        "Xgboost": XGBClassifier(eval_metric='logloss') #eval_metric='logloss': This tells XGBoost how to measure its own mistakes during training (Logarithmic Loss is standard for classification).
        ## XGBoost requires an explicit eval_metric like 'logloss' because it is a multi-purpose 
        # library that doesn't assume a default loss function like Logistic Regression does.
    }

    #variables to track best model
    best_model = None
    best_accuracy = 0
    best_f1 = 0
    best_auc = 0  # NEW: tracking best model using ROC-AUC
    best_run_id = None
    best_metrics = {}   # NEW: store full best model metrics

    #set mlflow experiments
    mlflow.set_experiment("Loan_Risk_Prediction")

    #loop through models
    for name, model in models.items():

        #start new mlflow run for each model
        with mlflow.start_run(run_name=name):

            mlflow.log_input(dataset, context="training")
            #train model
            model.fit(X_train,y_train)

            pred = model.predict(X_test)

            #calculate accuracy
            acc = accuracy_score(y_test,pred)
            precision = precision_score(y_test,pred)
            recall = recall_score(y_test,pred)
            f1 = f1_score(y_test,pred)

            #NEW: probability-based metrics (needed for ROC-AUC)
            try:
                prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, prob)
            except:
                auc = 0  # fallback for models without predict_proba

            #NEW: confusion matrix for deeper evaluation
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

            #log parameters
            mlflow.log_param("model_name",name)
            mlflow.log_param("dataset",'cleaned_loan_v1')
            
            #log special parameters
            if name == "Random_forest":
                mlflow.log_param("n_estimators",100)

            #log metrics
            mlflow.log_metric("accuracy",acc)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1_score",f1)
            mlflow.log_metric("roc_auc", auc)  # NEW metric

            #NEW: confusion matrix metrics logged separately
            mlflow.log_metric("true_negative", tn)
            mlflow.log_metric("false_positive", fp)
            mlflow.log_metric("false_negative", fn)
            mlflow.log_metric("true_positive", tp)

            #log model as artifact
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} accuracy: {acc} precision: {precision} recall: {recall} f1_score:{f1} roc_auc:{auc}")

            #track best model (UPDATED: now uses ROC-AUC first, fallback to F1)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_run_id = mlflow.active_run().info.run_id

                # NEW: store full metrics of best model
                best_metrics = {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": auc
                }

    print("Best model metrics:")
    print(best_metrics)

    return best_run_id, best_metrics


def register_model(run_id):

    """
    Registers the best model into MLflow Model Registry
    """

    # Create model URI using run_id
    model_uri = f"runs:/{run_id}/model"

    # Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name="loan_risk_model"   # Name in registry
    )

    # Print confirmation
    print(f"Model registered: {result.name}, version: {result.version}")


if __name__ == "__main__":

    # Step 1: Load data
    df = load_data()

    # Step 2: Train models and get best one
    run_id, best_metrics = train(df)

    print(f"Best model metrics: {best_metrics}")

    # Step 3: Register best model
    register_model(run_id)