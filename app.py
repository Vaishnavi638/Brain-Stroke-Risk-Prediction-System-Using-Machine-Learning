from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import joblib
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------------- DATABASE CONFIGURATION ----------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///stroke.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

app.secret_key = "your_secret_key_here"

# Load model
model_dict = joblib.load("model.joblib")
encoded_cols = model_dict["encoded_cols"]
numerical_cols = model_dict["numerical_cols"]
preprocessor = model_dict["preprocessor"]
clf_model = model_dict["model"]

# ---------------- DATABASE MODEL ----------------
class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # Personal Information
    full_name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    address = db.Column(db.String(200))
    mobile = db.Column(db.String(20))
    email = db.Column(db.String(100))
    smoking_status = db.Column(db.String(50))
    work_type = db.Column(db.String(50))
    ever_married = db.Column(db.String(10))

    # Medical Information
    bp = db.Column(db.Float)
    hypertension = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    avg_glucose_level = db.Column(db.Float)
    heart_disease = db.Column(db.Integer)
    residence_type = db.Column(db.String(50))

    # Prediction Information
    prediction = db.Column(db.String(50))
    probability = db.Column(db.String(100))

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PatientRecord {self.full_name}>"
    
# ---------------- CREATE DATABASE ----------------
with app.app_context():
    db.create_all()

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "becomp" and password == "1234":
            session["user"] = username
            return redirect(url_for("personal_info"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


# ---------------- PERSONAL INFO PAGE ----------------
@app.route("/personal_info", methods=["GET", "POST"])
def personal_info():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        # Store personal data in session
        session["full_name"] = request.form["full_name"]
        session["age"] = request.form["age"]
        session["gender"] = request.form["gender"]
        session["address"] = request.form["address"]
        session["mobile"] = request.form["mobile"]
        session["email"] = request.form["email"]
        session["smoking_status"] = request.form["smoking_status"]
        session["work_type"] = request.form["work_type"]
        session["ever_married"] = request.form["ever_married"]

        return redirect(url_for("medical_info"))

    return render_template("personal_info.html")


# ---------------- MEDICAL INFO PAGE ----------------
@app.route("/medical_info", methods=["GET", "POST"])
def medical_info():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        bp = float(request.form["bp"])

        # Hypertension: if BP >= 140 -> Yes (1), else No (0)
        hypertension = 1 if bp >= 140 else 0

        # Store BP in session
        session["bp"] = bp

        height = float(request.form["height"])
        weight = float(request.form["weight"])
        bmi = round(weight / ((height / 100) ** 2), 2)

        avg_glucose_level = float(request.form["avg_glucose_level"])
        heart_disease = int(request.form["heart_disease"])
        residence_type = request.form["residence_type"]

        # Store medical info in session
        session["hypertension"] = hypertension
        session["bmi"] = bmi
        session["heart_disease"] = heart_disease
        session["avg_glucose_level"] = avg_glucose_level
        session["Residence_type"] = residence_type

        # --- Normalize smoking status before prediction ---
        raw_smoke = session["smoking_status"].strip()
        smoking_map = {
            "Never Smoked": "never smoked",
            "Formerly Smoked": "formerly smoked",
            "Smokes": "smokes",
            "Unknown": "Unknown",
            "never smoked": "never smoked",
            "formerly smoked": "formerly smoked",
            "smokes": "smokes",
            "unknown": "Unknown"
        }
        smoking_status = smoking_map.get(raw_smoke, raw_smoke)

        single_input = {
            "gender": session["gender"],
            "age": float(session["age"]),
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": session["ever_married"],
            "work_type": session["work_type"],
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,  # ✅ fixed casing
        }


        # Preprocess and predict
        input_df = pd.DataFrame([single_input])
        encoded_array = preprocessor.transform(input_df)
        input_encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)
        input_numeric_df = input_df[numerical_cols].reset_index(drop=True)
        X = pd.concat([input_numeric_df, input_encoded_df], axis=1)

        pred = clf_model.predict(X)[0]
        prob = clf_model.predict_proba(X)[0][1]
        prob_percent = round(prob * 100, 2)

        if pred == 1:
            prediction = "Likely"
            probability = f"{prob_percent}% chance of stroke"
        else:
            prediction = "Not Likely"
            probability = f"{100 - prob_percent}% chance of no stroke"

        # --- Dashboard logic ---
        user_bmi = bmi
        user_glucose = avg_glucose_level
        safe_bmi_range = (18.5, 24.9)
        safe_glucose_range = (70, 140)

        safe_bp_lower = 90
        safe_bp_upper = 140

        if user_bmi < safe_bmi_range[0]:
            bmi_msg = "Your BMI is low. Please maintain a healthy diet."
        elif user_bmi > safe_bmi_range[1]:
            bmi_msg = "⚠️ Your BMI is high. Please lose weight for better health."
        else:
            bmi_msg = "✅ Your BMI is within the healthy range."

        if user_glucose < safe_glucose_range[0]:
            glucose_msg = "Your glucose level is low. Please consult a doctor."
        elif user_glucose > safe_glucose_range[1]:
            glucose_msg = "⚠️ Your glucose level is high. Please maintain your diet or see a doctor."
        else:
            glucose_msg = "✅ Your glucose level is within the safe range."
            
        # Blood Pressure message
        if bp < safe_bp_lower:
            bp_msg = "⚠️ Your blood pressure is low. Please consult a doctor."
        elif bp < safe_bp_upper:
            bp_msg = "✅ Your blood pressure is within the safe range."
        else:
            bp_msg = "⚠️ Your blood pressure is high. Please consult a doctor and monitor regularly."

                # --- Generate Chart ---
        plt.figure(figsize=(6, 4))
        categories = ["BMI", "Glucose", "Blood Pressure"]
        user_values = [user_bmi, user_glucose, bp]
        safe_upper = [safe_bmi_range[1], safe_glucose_range[1], safe_bp_upper]
        
        plt.bar(categories, user_values, color="orange", label="Your Value")
        plt.bar(categories, safe_upper, alpha=0.3, label="Safe Upper Limit")
        plt.title("Your Health Indicators vs Safe Range")
        plt.ylabel("Value")
        plt.legend(loc="upper left")

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # ✅ Save everything needed for the report
        session["prediction"] = prediction
        session["probability"] = probability
        session["bmi_msg"] = bmi_msg
        session["glucose_msg"] = glucose_msg
        session["bp_msg"] = bp_msg
        session["prob_percent"] = prob_percent
        session["bmi"] = user_bmi
        session["avg_glucose_level"] = user_glucose

                # ✅ Save everything needed for the report
        session["prediction"] = prediction
        session["probability"] = probability
        session["bmi_msg"] = bmi_msg
        session["glucose_msg"] = glucose_msg
        session["bp_msg"] = bp_msg
        session["prob_percent"] = prob_percent
        session["bmi"] = user_bmi
        session["avg_glucose_level"] = user_glucose

        # ---------------- SAVE RECORD TO DATABASE ----------------
        patient_record = PatientRecord(
            full_name=session.get("full_name"),
            age=int(session.get("age")),
            gender=session.get("gender"),
            address=session.get("address"),
            mobile=session.get("mobile"),
            email=session.get("email"),
            smoking_status=session.get("smoking_status"),
            work_type=session.get("work_type"),
            ever_married=session.get("ever_married"),

            bp=bp,
            hypertension=hypertension,
            bmi=bmi,
            avg_glucose_level=avg_glucose_level,
            heart_disease=heart_disease,
            residence_type=residence_type,

            prediction=prediction,
            probability=probability
        )

        db.session.add(patient_record)
        db.session.commit()

        # --- Render Combined Result + Dashboard Page ---
        return render_template(
        "result.html",
        prediction=prediction,
        probability=probability,
        prob_percent=prob_percent,
        plot_url=plot_url,
        bmi_msg=bmi_msg,
        glucose_msg=glucose_msg,
        bp_msg=bp_msg,
        user_bmi=user_bmi,
        user_glucose=user_glucose,
        user_bp=bp
    )

    
    return render_template("medical_info.html")

# ------------------ SUGGESTION PAGE ------------------
@app.route("/suggestion")
def suggestion():
    probability = request.args.get("probability", "0")
    try:
        probability_float = float(probability)
    except:
        probability_float = 0.0
    return render_template("suggestion.html", probability_float=probability_float) 


# ------------------ DOWNLOAD REPORT (FINAL FIXED VERSION) ------------------
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from flask import send_file
import io

@app.route("/download_report")
def download_report():
    if "user" not in session or "full_name" not in session:
        return redirect(url_for("login"))

    # Fetch all data safely from session
    full_name = session.get("full_name", "N/A")
    age = session.get("age", "N/A")
    gender = session.get("gender", "N/A")
    address = session.get("address", "N/A")
    mobile = session.get("mobile", "N/A")
    email = session.get("email", "N/A")
    smoking_status = session.get("smoking_status", "N/A")
    work_type = session.get("work_type", "N/A")
    ever_married = session.get("ever_married", "N/A")
    hypertension = session.get("hypertension", "N/A")
    heart_disease = session.get("heart_disease", "N/A")
    bmi = session.get("bmi", "N/A")
    glucose = session.get("avg_glucose_level", "N/A")
    bp = session.get("bp", "N/A")
    bp_msg = session.get("bp_msg", "N/A")
    residence = session.get("Residence_type", "N/A")

    # ✅ These now come from your latest session updates
    prediction = session.get("prediction", "N/A")
    probability = session.get("probability", "N/A")
    bmi_msg = session.get("bmi_msg", "N/A")
    glucose_msg = session.get("glucose_msg", "N/A")
    prob_percent = session.get("prob_percent", 0.0)

    # Suggestion logic (same as your app)
    if isinstance(prob_percent, str):
        try:
            prob_percent = float(prob_percent)
        except:
            prob_percent = 0.0

    if prob_percent >= 70:
        suggestion = [
            "🚨 High Risk of Stroke!",
            "Consult a neurologist immediately.",
            "Follow a low-salt, balanced diet with fruits & vegetables.",
            "Engage in regular light exercise.",
            "Take prescribed BP and cholesterol medicines regularly.",
            "Avoid smoking and alcohol consumption."
        ]
    elif prob_percent >= 40:
        suggestion = [
            "⚠️ Moderate Risk of Stroke!",
            "Maintain a healthy diet and regular exercise routine.",
            "Monitor blood pressure and glucose regularly.",
            "Stay hydrated and manage stress effectively.",
            "Avoid smoking and secondhand smoke."
        ]
    else:
        suggestion = [
            "✅ Low Risk of Stroke!",
            "Keep up your healthy lifestyle!",
            "Continue balanced diet, regular exercise, and proper sleep.",
            "Get periodic health checkups.",
        ]

    # --- Create PDF ---
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawCentredString(width / 2, y, "Brain Stroke Diagnosis Report")

    # Personal Info
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.darkred)
    c.drawString(50, y, "Personal Information:")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    y -= 20
    c.drawString(50, y, f"Name: {full_name}")
    y -= 20
    c.drawString(50, y, f"Age: {age}    Gender: {gender}")
    y -= 20
    c.drawString(50, y, f"Address: {address}")
    y -= 20
    c.drawString(50, y, f"Mobile: {mobile}    Email: {email}")
    y -= 20
    c.drawString(50, y, f"Smoking Status: {smoking_status}")
    y -= 20
    c.drawString(50, y, f"Work Type: {work_type}    Ever Married: {ever_married}")

    # Medical Info
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.darkred)
    c.drawString(50, y, "Medical Information:")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    y -= 20
    c.drawString(50, y, f"Hypertension: {hypertension}")
    y -= 20
    c.drawString(50, y, f"Heart Disease: {heart_disease}")
    y -= 20
    c.drawString(50, y, f"BMI: {bmi}")
    y -= 20
    c.drawString(50, y, f"Average Glucose Level: {glucose}")
    y -= 20
    c.drawString(50, y, f"Blood Pressure: {bp}")
    y -= 20
    c.drawString(50, y, f"Residence Type: {residence}")

    # Stroke Prediction Result
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.darkgreen)
    c.drawString(50, y, "Stroke Prediction Result:")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    y -= 20
    c.drawString(50, y, f"Prediction: {prediction}")
    y -= 20
    c.drawString(50, y, f"Probability: {probability}")

    # Health Dashboard Summary
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.darkblue)
    c.drawString(50, y, "Health Dashboard Summary:")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    y -= 20
    c.drawString(50, y, f"BMI Message: {bmi_msg}")
    y -= 20
    c.drawString(50, y, f"Glucose Message: {glucose_msg}")
    y -= 20
    c.drawString(50, y, f"Blood Pressure Message: {bp_msg}")

    # Emergency and Medical Suggestions
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.red)
    c.drawString(50, y, "Emergency & Medical Suggestions:")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    for item in suggestion:
        y -= 18
        c.drawString(70, y, f"- {item}")
        if y < 100:
            c.showPage()
            y = height - 50

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="Brain_Stroke_Report.pdf", mimetype="application/pdf")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- HOME REDIRECT ----------------
@app.route("/")
def home_redirect():
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)

