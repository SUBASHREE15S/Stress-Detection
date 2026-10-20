from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import os

app = Flask(__name__)

# ---------------- CREATE STATIC FOLDER IF MISSING ----------------
if not os.path.exists("static"):
    os.makedirs("static")

# ---------------- GLOBAL STORAGE FOR PDF ----------------
last_prediction = {}

# ---------------- LOAD DATASET ----------------
csv_path = r"C:\Users\srimu\Downloads\mobile_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}")

df = pd.read_csv(csv_path)
df.dropna(inplace=True)

# ---------------- FEATURES ----------------
features = [
    'Daily_Screen_Time_Hours',
    'Phone_Unlocks_Per_Day',
    'Social_Media_Usage_Hours',
    'Gaming_Usage_Hours',
    'Streaming_Usage_Hours',
    'Messaging_Usage_Hours',
    'Work_Related_Usage_Hours',
    'Sleep_Hours',
    'Physical_Activity_Hours',
    'Time_Spent_With_Family_Hours',
    'Online_Shopping_Hours',
    'Push_Notifications_Per_Day'
]

df = df[features]

# ---------------- STRESS SCORE ----------------
df['Stress_Score'] = (
    df['Daily_Screen_Time_Hours'] * 2 +
    df['Phone_Unlocks_Per_Day'] * 0.3 +
    df['Social_Media_Usage_Hours'] * 1.5 +
    df['Gaming_Usage_Hours'] * 1.5 +
    df['Streaming_Usage_Hours'] * 1.2 +
    df['Work_Related_Usage_Hours'] * 1.3 +
    df['Push_Notifications_Per_Day'] * 0.2 -
    df['Sleep_Hours'] * 4 -
    df['Physical_Activity_Hours'] * 3 -
    df['Time_Spent_With_Family_Hours'] * 2
)

# ---------------- STRESS CLASS ----------------
low = df['Stress_Score'].quantile(0.33)
high = df['Stress_Score'].quantile(0.66)

def label_stress(x):
    if x <= low:
        return 0
    elif x <= high:
        return 1
    else:
        return 2

df['Stress_Class'] = df['Stress_Score'].apply(label_stress)

X = df[features]
y_class = df['Stress_Class']
y_reg = df['Stress_Score']

# ---------------- BALANCE DATA ----------------
sm = SMOTE(random_state=42)
X_res, y_class_res = sm.fit_resample(X, y_class)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_class_res, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODELS ----------------
clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
clf.fit(X_train, y_train)

reg = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
reg.fit(X_train_r, y_train_r)

# ---------------- EVALUATION ----------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# ---------------- SAVE CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()

# ---------------- REGRESSION METRICS ----------------
y_pred_r = reg.predict(X_test_r)
rmse = mean_squared_error(y_test_r, y_pred_r) ** 0.5

mae = mean_absolute_error(y_test_r, y_pred_r)

# ---------------- FEATURE IMPORTANCE ----------------
importances = clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="#4caf50")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html",
                           accuracy=round(accuracy * 100, 2),
                           rmse=round(rmse, 2),
                           mae=round(mae, 2))

@app.route("/predict", methods=["POST"])
def predict():
    global last_prediction

    data = request.get_json(force=True)
    input_df = pd.DataFrame([{f: float(data.get(f, 0)) for f in features}])

    class_pred = clf.predict(input_df)[0]
    probs = clf.predict_proba(input_df)[0]
    score = reg.predict(input_df)[0]

    stress_map = {0: "Low", 1: "Medium", 2: "High"}

    last_prediction = {
        "stress_level": stress_map[class_pred],
        "stress_score": round(float(score), 2),
        "probabilities": {
            "Low": round(probs[0] * 100, 2),
            "Medium": round(probs[1] * 100, 2),
            "High": round(probs[2] * 100, 2)
        }
    }

    return jsonify(last_prediction)

# ---------------- PDF REPORT ----------------
@app.route("/download_report")
def download_report():
    if not last_prediction:
        return "No prediction available", 400

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Stress Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 760, f"Stress Level: {last_prediction['stress_level']}")
    c.drawString(100, 740, f"Stress Score: {last_prediction['stress_score']}")

    y = 700
    c.drawString(100, y, "Prediction Probabilities:")
    y -= 20

    for k, v in last_prediction["probabilities"].items():
        c.drawString(120, y, f"{k}: {v}%")
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="stress_report.pdf",
        mimetype="application/pdf"
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000/")
    app.run(debug=True)
