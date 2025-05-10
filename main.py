from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pycaret.classification import setup, compare_models, pull
import pandas as pd
import joblib
import os
import uuid


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

app = FastAPI()

# Authorise CORS for use with Shiny
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour test local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder for saving models
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Mapping of supervised models
model_map = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "svm": SVC
}


@app.post("/automl")
async def automl(file: UploadFile = File(...), target: str = Form(...)):
    df = pd.read_csv(file.file)
    
    clf = setup(data=df, target=target, session_id=123)
    best_model = compare_models(n_select=10)  # ← gets the 10 best models
    leaderboard_df = pull()  # ← retrieves the full array after compare_models()

    return {
        "best_model_name": str(best_model),
        "leaderboard": leaderboard_df.to_dict(orient="records")  # ← full leaderboard
    }



@app.post("/train_sklearn")
async def train_sklearn_model(
    file: UploadFile,
    target: str = Form(...),
    features: str = Form(...),
    model_type: str = Form(...),
    test_size: float = Form(0.2)
):
    df = pd.read_csv(file.file)
    feature_list = features.split(",")
    
    X = df[feature_list]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    model_cls = model_map.get(model_type)
    if model_cls is None:
        return {"error": f"Modèle '{model_type}' not supported"}

    model = model_cls()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)

    # Save model
    model_id = str(uuid.uuid4())[:8]
    model_name = f"{model_type}_{model_id}.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)
    joblib.dump(model, model_path)

    return {
        "model_name": model_name,
        "report": report
    }


@app.post("/predict-custom")
async def predict_custom(
    file: UploadFile,
    model_name: str = Form(...)
):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return {"error": "Modèle introuvable."}

    model = joblib.load(model_path)
    df = pd.read_csv(file.file)
    predictions = model.predict(df)

    return {
        "predictions": predictions.tolist()
    }


@app.post("/run_unsupervised")
async def train_unsupervised_model(
    file: UploadFile,
    model_type: str = Form(...),
    n_clusters: int = Form(None),
    n_components: int = Form(2)
):
    df = pd.read_csv(file.file)
    df_numeric = df.select_dtypes(include=["number"]).dropna()

    if df_numeric.shape[0] == 0:
        return {"error": "No valid numeric columns."}

    metrics = ""
    plot_data = {}

    if model_type == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_numeric)
        labels = kmeans.labels_
        metrics = f"Inertia (distortion): {kmeans.inertia_}"

        # PCA pour visualisation 2D
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(df_numeric)

        plot_data = {
            "x": reduced[:, 0].tolist(),
            "y": reduced[:, 1].tolist(),
            "labels": labels.tolist()
        }

    elif model_type == "pca":
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(df_numeric)
        explained = pca.explained_variance_ratio_
        metrics = f"Variance explained: {explained.tolist()}"

        plot_data = {
            "x": reduced[:, 0].tolist(),
            "y": reduced[:, 1].tolist(),
            "labels": [0] * len(reduced)  # Une seule classe pour visualiser
        }

    else:
        return {"error": f"Unsupervised model unknown: {model_type}"}

    return {
        "metrics": metrics,
        "plot_data": plot_data
    }













