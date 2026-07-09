import json
import os
import re
import uuid
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

# Required when Flask runs without a desktop GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, url_for
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from flask_cors import CORS


UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
STORAGE_FOLDER = "storage"

os.makedirs(...)

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"

# =====================================================
# Flask Configuration
# =====================================================

app = Flask(__name__)
CORS(app)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"

DATASET_PATH = DATASET_DIR / "current_dataset.csv"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# NLTK Setup
# =====================================================

def ensure_nltk_resource(resource_path, package_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name, quiet=True)


ensure_nltk_resource("corpora/stopwords", "stopwords")
ensure_nltk_resource("corpora/wordnet", "wordnet")
ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")

all_stopwords = set(stopwords.words("english"))
all_stopwords.discard("not")

lemmatizer = WordNetLemmatizer()


# =====================================================
# Global Dataset Cache
# =====================================================

df = None
loaded_file_mtime = None


# =====================================================
# Custom Error
# =====================================================

class DatasetError(Exception):
    pass


# =====================================================
# JSON Helper
# =====================================================

def dataframe_records(dataframe):
    """
    Convert Pandas DataFrame into JSON-safe Python records.
    """
    return json.loads(dataframe.to_json(orient="records"))


# =====================================================
# Dataset Loading
# =====================================================

def read_dataset(csv_path):
    """
    Read and validate CSV uploaded through upload.php.
    Required columns:
    - Review
    - Rating
    """

    try:
        data = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, encoding="latin-1")

    # Normalize potential BOM / whitespace in CSV headers.
    data.columns = (
        data.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # Optional unused columns.
    data.drop(columns=["Title", "Date"], errors="ignore", inplace=True)

    required_columns = {"Review", "Rating"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise DatasetError(
            "CSV wajib memiliki kolom: "
            + ", ".join(sorted(missing_columns))
        )

    data.dropna(subset=["Review", "Rating"], inplace=True)

    data["Review"] = data["Review"].astype(str)
    data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce")

    data.dropna(subset=["Rating"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    if data.empty:
        raise DatasetError("Dataset kosong setelah validasi.")

    return data


def get_dataset():
    """
    Use cached data when possible.
    Reload current_dataset.csv automatically if Flask restarts
    or a newer dataset is uploaded.
    """

    global df, loaded_file_mtime

    if not DATASET_PATH.exists():
        df = None
        loaded_file_mtime = None
        return None

    current_mtime = DATASET_PATH.stat().st_mtime_ns

    if df is None or loaded_file_mtime != current_mtime:
        df = read_dataset(DATASET_PATH)
        loaded_file_mtime = current_mtime

        print(
            f"[DATASET LOADED] {len(df)} rows | "
            f"source: {DATASET_PATH.name}"
        )

    return df


# =====================================================
# Text Cleaning
# =====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    return str(text).split()


def lemmatize_text(tokens):
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in all_stopwords
    )


# =====================================================
# Preprocessing and Labeling
# =====================================================

def preprocess_dataset():
    source_data = get_dataset()

    if source_data is None:
        return None

    data = source_data.copy()

    data["cleaned_text"] = data["Review"].apply(clean_text)
    data["tokens"] = data["cleaned_text"].apply(tokenize_text)
    data["lemmatized_review"] = data["tokens"].apply(lemmatize_text)

    # Rating 1–5 = Negative
    # Rating 7–10 = Positive
    # Rating 6 = ignored
    data = data[
        (data["Rating"] <= 5) | (data["Rating"] >= 7)
    ].copy()

    data["label"] = data["Rating"].apply(
        lambda rating: 1 if rating >= 7 else 0
    )

    return data


def require_preprocessed_dataset():
    data = preprocess_dataset()

    if data is None:
        return None, (
            jsonify({
                "success": False,
                "message": "Silakan upload dataset terlebih dahulu."
            }),
            400
        )

    if data.empty:
        return None, (
            jsonify({
                "success": False,
                "message": (
                    "Tidak ada data yang dapat diproses. "
                    "Pastikan CSV memiliki Rating 1–5 atau 7–10."
                )
            }),
            400
        )

    return data, None


# =====================================================
# Train / Test Split
# =====================================================

def split_dataset(data=None):
    if data is None:
        data = preprocess_dataset()

    if data is None or data.empty:
        raise DatasetError("Dataset belum tersedia.")

    X = data["lemmatized_review"]
    y = data["label"]

    label_counts = y.value_counts()

    if len(label_counts) < 2:
        raise DatasetError(
            "Dataset harus memiliki data sentimen positif dan negatif."
        )

    if label_counts.min() < 2:
        raise DatasetError(
            "Setiap kelas sentimen minimal membutuhkan 2 data."
        )

    if len(data) < 10:
        raise DatasetError(
            "Dataset terlalu kecil. Minimal gunakan 10 data."
        )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =====================================================
# Model Training and Evaluation
# =====================================================

def train_and_evaluate(data=None):
    if data is None:
        data = preprocess_dataset()

    X_train, X_test, y_train, y_test = split_dataset(data)

    tfidf = TfidfVectorizer(
        max_df=0.5,
        min_df=2
    )

    try:
        X_train_vect = tfidf.fit_transform(X_train)
        X_test_vect = tfidf.transform(X_test)
    except ValueError as error:
        raise DatasetError(
            f"TF-IDF gagal dibuat: {str(error)}"
        ) from error

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vect, y_train)

    prediction = model.predict(X_test_vect)

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    # Save active model and vectorizer.
    joblib.dump(model, STORAGE_DIR / "model.joblib")
    joblib.dump(tfidf, STORAGE_DIR / "vectorizer.joblib")

    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(
            precision_score(
                y_test,
                prediction,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                prediction,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                prediction,
                zero_division=0
            )
        ),
        "train_total": int(len(X_train)),
        "test_total": int(len(X_test)),
        "feature_total": int(
            len(tfidf.get_feature_names_out())
        ),
        "matrix": {
            "tp": int(cm[1][1]),
            "fn": int(cm[1][0]),
            "fp": int(cm[0][1]),
            "tn": int(cm[0][0])
        }
    }

    return metrics, cm


# =====================================================
# Summary
# =====================================================

def build_summary(data=None):
    if data is None:
        data = preprocess_dataset()

    if data is None:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0
        }

    return {
        "total": int(len(data)),
        "positive": int((data["label"] == 1).sum()),
        "negative": int((data["label"] == 0).sum())
    }


# =====================================================
# Word Cloud
# =====================================================

def save_wordcloud(text, file_name, title, color_map):
    if not text.strip():
        image_path = STATIC_DIR / file_name

        if image_path.exists():
            image_path.unlink()

        return None

    wordcloud = WordCloud(
        background_color="white",
        colormap=color_map,
        width=900,
        height=500
    ).generate(text)

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.imshow(wordcloud)
    axis.axis("off")
    axis.set_title(title)

    figure.tight_layout()
    figure.savefig(
        STATIC_DIR / file_name,
        bbox_inches="tight"
    )

    plt.close(figure)

    return url_for(
        "static",
        filename=file_name,
        _external=True
    )


def generate_wordclouds(data):
    positive_text = " ".join(
        data[data["label"] == 1]["lemmatized_review"]
        .astype(str)
        .tolist()
    )

    negative_text = " ".join(
        data[data["label"] == 0]["lemmatized_review"]
        .astype(str)
        .tolist()
    )

    positive_url = save_wordcloud(
        positive_text,
        "positive_wordcloud.png",
        "Positive Reviews Word Cloud",
        "Greens"
    )

    negative_url = save_wordcloud(
        negative_text,
        "negative_wordcloud.png",
        "Negative Reviews Word Cloud",
        "Reds"
    )

    return {
        "positive": positive_url,
        "negative": negative_url
    }


# =====================================================
# Confusion Matrix Image
# =====================================================

def save_confusion_matrix(cm):
    labels = ["Negative", "Positive"]

    figure, axis = plt.subplots(figsize=(6, 5))

    image = axis.imshow(cm, cmap="Blues")

    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")

    axis.set_xticks(np.arange(2))
    axis.set_yticks(np.arange(2))

    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)

    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            axis.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
                color="black",
                fontsize=14
            )

    figure.colorbar(image)
    figure.tight_layout()

    file_name = "confusion_matrix.png"

    figure.savefig(
        STATIC_DIR / file_name,
        bbox_inches="tight"
    )

    plt.close(figure)

    return url_for(
        "static",
        filename=file_name,
        _external=True
    )


# =====================================================
# Top Words
# =====================================================

def get_top_words(data, label, top_n=20):
    subset = data[
        data["label"] == label
    ]["lemmatized_review"]

    counter = Counter()

    for text in subset:
        counter.update(str(text).split())

    return counter.most_common(top_n)


def save_top_words_chart(words_data, file_name, title):
    words, counts = (
        zip(*words_data)
        if words_data
        else ([], [])
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.barh(
        list(words)[::-1],
        list(counts)[::-1]
    )

    axis.set_title(title)
    axis.set_xlabel("Frekuensi")

    figure.tight_layout()

    figure.savefig(
        STATIC_DIR / file_name,
        bbox_inches="tight"
    )

    plt.close(figure)

    return url_for(
        "static",
        filename=file_name,
        _external=True
    )


# =====================================================
# Full Predictions
# =====================================================

def get_full_predictions(data):
    X_train, _, y_train, _ = split_dataset(data)

    tfidf = TfidfVectorizer(
        max_df=0.5,
        min_df=2
    )

    try:
        X_train_vect = tfidf.fit_transform(X_train)
        X_all_vect = tfidf.transform(
            data["lemmatized_review"]
        )
    except ValueError as error:
        raise DatasetError(
            f"TF-IDF gagal dibuat: {str(error)}"
        ) from error

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vect, y_train)

    predictions = model.predict(X_all_vect)

    rows = []

    for (_, row), prediction in zip(
        data.iterrows(),
        predictions
    ):
        actual_label = int(row["label"])
        predicted_label = int(prediction)

        rows.append({
            "review": str(row["Review"]),
            "actual": (
                "Positif"
                if actual_label == 1
                else "Negatif"
            ),
            "predicted": (
                "Positif"
                if predicted_label == 1
                else "Negatif"
            ),
            "status": (
                "Benar"
                if actual_label == predicted_label
                else "Salah"
            )
        })

    return rows


# =====================================================
# Error Handler
# =====================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    return jsonify({
        "success": False,
        "message": "Ukuran file terlalu besar. Maksimal 10 MB."
    }), 413


# =====================================================
# Basic Routes
# =====================================================

@app.route("/")
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "status": "running",
        "dataset_exists": DATASET_PATH.exists()
    })


# =====================================================
# Upload Route
# Called by upload.php through cURL
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():
    global df, loaded_file_mtime

    file = request.files.get("dataset")

    if file is None or file.filename == "":
        return jsonify({
            "success": False,
            "message": "File dataset tidak ditemukan."
        }), 400

    file_name = secure_filename(file.filename)

    if not file_name.lower().endswith(".csv"):
        return jsonify({
            "success": False,
            "message": "File harus berformat CSV."
        }), 400

    temporary_path = DATASET_DIR / (
        f"temporary_{uuid.uuid4().hex}.csv"
    )

    try:
        # Save temporary file first.
        file.save(temporary_path)

        # Validate before replacing current active dataset.
        uploaded_data = read_dataset(temporary_path)

        # Replace old dataset only after validation succeeds.
        os.replace(temporary_path, DATASET_PATH)

        df = uploaded_data
        loaded_file_mtime = DATASET_PATH.stat().st_mtime_ns

        return jsonify({
            "success": True,
            "message": "Dataset berhasil diproses oleh Flask.",
            "filename": file_name,
            "rows": int(len(df))
        }), 200

    except DatasetError as error:
        if temporary_path.exists():
            temporary_path.unlink()

        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()

        return jsonify({
            "success": False,
            "message": f"Gagal membaca CSV: {str(error)}"
        }), 500


# =====================================================
# Dashboard
# =====================================================

@app.route("/dashboard")
def dashboard():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return jsonify({
            "uploaded": False,
            "total": 0,
            "positive": 0,
            "negative": 0,
            "preview": [],
            "message": "Silakan upload dataset terlebih dahulu."
        })

    preview = data[["Rating", "Review"]].head(8)

    return jsonify({
        "uploaded": True,
        "total": int(len(data)),
        "positive": int((data["label"] == 1).sum()),
        "negative": int((data["label"] == 0).sum()),
        "preview": dataframe_records(preview)
    })


# =====================================================
# Labeling
# =====================================================

@app.route("/labeling")
def labeling():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    preview = data[
        ["Rating", "Review", "label"]
    ].head(10)

    preview["sentiment"] = preview["label"].apply(
        lambda value: (
            "Positive"
            if value == 1
            else "Negative"
        )
    )

    return jsonify({
        "success": True,
        "total": int(len(data)),
        "positive": int((data["label"] == 1).sum()),
        "negative": int((data["label"] == 0).sum()),
        "rows": dataframe_records(preview)
    })


# =====================================================
# Preprocessing
# =====================================================

@app.route("/preprocess")
def preprocess():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    preview = data[
        ["Review", "lemmatized_review", "label"]
    ].head(15).copy()

    preview["sentiment"] = preview["label"].apply(
        lambda value: (
            "Positive"
            if value == 1
            else "Negative"
        )
    )

    preview.rename(columns={
        "Review": "review",
        "lemmatized_review": "clean_review"
    }, inplace=True)

    preview.drop(columns=["label"], inplace=True)

    return jsonify({
        "success": True,
        "total": int(len(data)),
        "rows": dataframe_records(preview)
    })


# =====================================================
# Dataset Split
# =====================================================

@app.route("/split")
def split():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    try:
        X_train, X_test, y_train, y_test = split_dataset(data)

        return jsonify({
            "success": True,
            "total": int(len(data)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "train_percent": 80,
            "test_percent": 20,
            "random_state": 42,
            "train_positive": int((y_train == 1).sum()),
            "train_negative": int((y_train == 0).sum()),
            "test_positive": int((y_test == 1).sum()),
            "test_negative": int((y_test == 0).sum())
        })

    except DatasetError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400


# =====================================================
# Training
# =====================================================

@app.route("/train")
def train():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    try:
        metrics, _ = train_and_evaluate(data)

        return jsonify({
            "success": True,
            **metrics
        })

    except DatasetError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400


# =====================================================
# Full Analysis
# =====================================================

@app.route("/analysis")
def analysis():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    try:
        metrics, cm = train_and_evaluate(data)

        wordcloud_urls = generate_wordclouds(data)
        confusion_url = save_confusion_matrix(cm)

        return jsonify({
            "success": True,
            "summary": build_summary(data),
            "metrics": metrics,
            "wordcloud": wordcloud_urls,
            "confusion": {
                "image": confusion_url
            }
        })

    except DatasetError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400


# =====================================================
# Word Cloud
# =====================================================

@app.route("/wordcloud")
def wordcloud():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    urls = generate_wordclouds(data)

    return jsonify({
        "success": True,
        **urls
    })


# =====================================================
# Confusion Matrix
# =====================================================

@app.route("/confusion")
def confusion():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    try:
        metrics, cm = train_and_evaluate(data)
        image_url = save_confusion_matrix(cm)

        return jsonify({
            "success": True,
            "matrix": metrics["matrix"],
            "image": image_url
        })

    except DatasetError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400


# =====================================================
# Top Words
# =====================================================

@app.route("/top-words")
def top_words():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    try:
        top_n = int(request.args.get("top", 20))
        top_n = max(1, min(top_n, 100))
    except ValueError:
        top_n = 20

    positive_words = get_top_words(data, 1, top_n)
    negative_words = get_top_words(data, 0, top_n)

    positive_chart = save_top_words_chart(
        positive_words,
        "top_words_positive.png",
        f"Top {top_n} Kata - Sentimen Positif"
    )

    negative_chart = save_top_words_chart(
        negative_words,
        "top_words_negative.png",
        f"Top {top_n} Kata - Sentimen Negatif"
    )

    return jsonify({
        "success": True,
        "top": top_n,
        "positive": [
            {"word": word, "count": count}
            for word, count in positive_words
        ],
        "negative": [
            {"word": word, "count": count}
            for word, count in negative_words
        ],
        "charts": {
            "positive": positive_chart,
            "negative": negative_chart,
            "wordcloud_positive": url_for(
                "static",
                filename="positive_wordcloud.png",
                _external=True
            ),
            "wordcloud_negative": url_for(
                "static",
                filename="negative_wordcloud.png",
                _external=True
            )
        }
    })


# =====================================================
# Predictions
# =====================================================

@app.route("/predictions")
def predictions():
    data, error_response = require_preprocessed_dataset()

    if error_response:
        return error_response

    try:
        rows = get_full_predictions(data)

        status_filter = request.args.get("status")
        actual_filter = request.args.get("actual")
        predicted_filter = request.args.get("predicted")

        if status_filter:
            rows = [
                row for row in rows
                if row["status"] == status_filter
            ]

        if actual_filter:
            rows = [
                row for row in rows
                if row["actual"] == actual_filter
            ]

        if predicted_filter:
            rows = [
                row for row in rows
                if row["predicted"] == predicted_filter
            ]

        correct = sum(
            1
            for row in rows
            if row["status"] == "Benar"
        )

        wrong = len(rows) - correct

        return jsonify({
            "success": True,
            "total": int(len(rows)),
            "correct": int(correct),
            "wrong": int(wrong),
            "accuracy": round(
                (correct / len(rows)) * 100,
                2
            ) if rows else 0,
            "rows": rows
        })

    except DatasetError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

# =====================================================
# Distribution Chart
# =====================================================
 
@app.route("/distribution-chart")
def distribution_chart():
    data, error_response = require_preprocessed_dataset()
 
    if error_response:
        return error_response
 
    summary = build_summary(data)
 
    total     = summary["total"]
    positive  = summary["positive"]
    negative  = summary["negative"]
    pos_pct   = round(positive / total * 100, 2) if total else 0
    neg_pct   = round(negative / total * 100, 2) if total else 0
 
    # Generate pie chart
    labels = ["Positif", "Negatif"]
    values = [positive, negative]
    colors = ["#2e7d32", "#c62828"]
 
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )
    axis.set_title("Distribusi Sentimen")
    figure.tight_layout()
    figure.savefig(STATIC_DIR / "distribution_pie.png", bbox_inches="tight")
    plt.close(figure)
 
    chart_url = url_for(
        "static",
        filename="distribution_pie.png",
        _external=True
    )
 
    return jsonify({
        "success": True,
        "summary": summary,
        "positive_percent": pos_pct,
        "negative_percent": neg_pct,
        "chart": chart_url
    })

@app.route("/reset", methods=["POST"])
def reset():
    global df, loaded_file_mtime

    # kosongkan cache
    df = None
    loaded_file_mtime = None

    # hapus dataset aktif
    if DATASET_PATH.exists():
        DATASET_PATH.unlink()

    # hapus model
    model_file = STORAGE_DIR / "model.joblib"
    if model_file.exists():
        model_file.unlink()

    # hapus vectorizer
    vectorizer_file = STORAGE_DIR / "vectorizer.joblib"
    if vectorizer_file.exists():
        vectorizer_file.unlink()

    # hapus semua gambar hasil analisis
    images = [
        "positive_wordcloud.png",
        "negative_wordcloud.png",
        "confusion_matrix.png",
        "distribution_pie.png",
        "top_words_positive.png",
        "top_words_negative.png"
    ]

    for image in images:
        path = STATIC_DIR / image
        if path.exists():
            path.unlink()

    return jsonify({
        "success": True,
        "message": "Reset berhasil."
    })


# =====================================================
# Start Flask
# =====================================================

# if __name__ == "__main__":
#     app.run(
#         host="127.0.0.1",
#         port=5000,
#         debug=True
#     )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(
        host="0.0.0.0",
        port=port
    )