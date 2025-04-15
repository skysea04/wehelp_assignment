from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.serialization
import torch.nn.functional as F
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec
from tokenizer import tokenize
from constants import BOARDS
from title_classification_model_trainer import TitleClassifier
from contextlib import asynccontextmanager

# Constants
PREDICTIONS_FILE = "predictions.csv"


class MLModels:
    embedding_model: Doc2Vec | None = None
    classifier: TitleClassifier | None = None


ml_models = MLModels()


def load_models():
    embedding_model_path = "data/embedding_model_30_2_5_10_1_0_0.d2v"
    embedding_model: Doc2Vec = Doc2Vec.load(embedding_model_path)

    model_path = "data/title_classifier_30_1200_1200_9_1000000.pth"
    classifier = TitleClassifier(
        input_size=30,
        hidden_sizes=[1200, 1200],
        num_classes=9,
    )
    classifier.load_state_dict(torch.load(model_path)["model_state_dict"])
    classifier.eval()

    return embedding_model, classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    embedding_model, classifier = load_models()
    ml_models.embedding_model = embedding_model
    ml_models.classifier = classifier

    # Create predictions CSV file if it doesn't exist
    if not Path(PREDICTIONS_FILE).exists():
        pd.DataFrame(
            columns=["timestamp", "title", "distribution", "prediction", "feedback"]
        ).to_csv(PREDICTIONS_FILE, index=False)

    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "boards": BOARDS}
    )


@app.post("/predict")
async def predict(title: str = Form(...)):
    # Get current timestamp in milliseconds
    timestamp = int(datetime.now().timestamp() * 1000)

    # Tokenize the title
    tokenized = tokenize(["dummy"], [title])[0].split(",")[1:]

    # Implement embedding and prediction
    embedding = ml_models.embedding_model.infer_vector(tokenized)
    distribution = ml_models.classifier(torch.FloatTensor(embedding))
    prediction = BOARDS[torch.argmax(distribution)]

    # Convert tensor to list
    distribution_list = [round(x, 3) for x in F.softmax(distribution, dim=0).tolist()]

    # Save to CSV
    df = pd.read_csv(PREDICTIONS_FILE)
    new_row = {
        "timestamp": timestamp,
        "title": title,
        "distribution": json.dumps(distribution_list),
        "prediction": prediction,
        "feedback": None,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(PREDICTIONS_FILE, index=False)

    return JSONResponse(
        {
            "timestamp": timestamp,
            "title": title,
            "distribution": distribution_list,
            "prediction": prediction,
        }
    )


@app.post("/feedback")
async def feedback(request: Request):
    data = await request.json()
    timestamp = data.get("timestamp")
    title = data.get("title")
    feedback_value = data.get("feedback")

    # Update CSV
    df = pd.read_csv(PREDICTIONS_FILE)
    mask = (df["timestamp"] == timestamp) & (df["title"] == title)
    if not mask.any():
        return JSONResponse({"ok": False, "error": "Record not found"})

    df.loc[mask, "feedback"] = feedback_value
    df.to_csv(PREDICTIONS_FILE, index=False)

    return JSONResponse({"ok": True})
