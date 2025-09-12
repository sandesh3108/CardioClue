import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from dotenv import load_dotenv
from pymongo import MongoClient

from utils.otp import generate_otp
from utils.pdf import generate_report_pdf
from utils.risk import compute_risk_level


load_dotenv()


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key_change_me")

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db_name = os.getenv("MONGODB_DB", "cardioclue")
    db = client[db_name]

    users = db.users
    tests = db.tests

    # -----------------------------
    # Helpers
    # -----------------------------
    def require_login():
        if "userId" not in session:
            return False
        return True

    def is_admin():
        return session.get("role") == "admin"

    def is_valid_phone(phone: str):
        return phone.isdigit() and len(phone) == 10

    # -----------------------------
    # Routes
    # -----------------------------
    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            # Check if it's admin login
            admin_name = request.form.get("admin_name", "").strip()
            admin_phone = request.form.get("admin_phone", "").strip()
            
            if admin_name and admin_phone:
                if not is_valid_phone(admin_phone):
                    flash("Admin phone must be exactly 10 digits.", "danger")
                    return render_template("login.html")
                # Admin login flow
                admin_user = users.find_one({"name": admin_name, "phone": admin_phone, "role": "admin"})
                if admin_user:
                    session["userId"] = admin_user["userId"]
                    session["name"] = admin_user.get("name")
                    session["role"] = "admin"
                    session["age"] = admin_user.get("age")
                    session["gender"] = admin_user.get("gender")
                    flash("Admin login successful!", "success")
                    return redirect(url_for("admin"))
                else:
                    flash("Invalid admin credentials.", "danger")
                    return render_template("login.html")
            
            # Regular user login flow
            name = request.form.get("name", "").strip()
            phone = request.form.get("phone", "").strip()
            if not name or not phone:
                flash("Please provide both name and phone number.", "danger")
                return render_template("login.html")
            if not is_valid_phone(phone):
                flash("Phone number must be exactly 10 digits.", "danger")
                return render_template("login.html")

            code = generate_otp()
            session["pending_name"] = name
            session["pending_phone"] = phone
            session["otp_code"] = code
            app.logger.info(f"Simulated OTP for {phone}: {code}")
            return redirect(url_for("verify"))

        return render_template("login.html")

    @app.route("/verify", methods=["GET", "POST"])
    def verify():
        pending_name = session.get("pending_name")
        pending_phone = session.get("pending_phone")
        code = session.get("otp_code")
        if not (pending_name and pending_phone and code):
            return redirect(url_for("index"))

        if request.method == "POST":
            otp_entered = request.form.get("otp", "").strip()
            if otp_entered != str(code):
                flash("Invalid OTP. Please try again.", "danger")
                return render_template("verify.html", otp=code, name=pending_name, phone=pending_phone)

            # OTP verified
            existing = users.find_one({"name": pending_name, "phone": pending_phone})
            if existing:
                session["userId"] = existing["userId"]
                session["name"] = existing.get("name")
                session["role"] = existing.get("role", "user")
                session["age"] = existing.get("age")
                session["gender"] = existing.get("gender")
                flash("Welcome back!", "success")
                if existing.get("role") == "admin":
                    return redirect(url_for("admin"))
                return redirect(url_for("user_dashboard", userId=existing["userId"]))

            # Create new user (collect basic info next)
            new_user_id = str(uuid.uuid4())
            session["userId"] = new_user_id
            session["name"] = pending_name
            session["role"] = "user"
            flash("OTP verified. Please provide your basic details.", "info")
            return redirect(url_for("onboarding"))

        return render_template("verify.html", otp=code, name=pending_name, phone=pending_phone)

    @app.route("/admin")
    def admin():
        if not require_login():
            return redirect(url_for("index"))
        if not is_admin():
            flash("Access denied.", "danger")
            return redirect(url_for("user_dashboard", userId=session.get("userId")))

        all_users = list(users.find({}, {"_id": 0}))
        # Attach latest test risk
        for u in all_users:
            latest = tests.find_one({"userId": u["userId"]}, sort=[("timestamp", -1)], projection={"_id": 0, "riskResult": 1})
            u["latestRisk"] = latest.get("riskResult") if latest else None

        return render_template("admin.html", users=all_users)

    @app.route("/onboarding", methods=["GET", "POST"])
    def onboarding():
        if not require_login():
            return redirect(url_for("index"))

        user_id = session.get("userId")
        # If user already exists with age/gender, skip onboarding
        existing = users.find_one({"userId": user_id}, {"_id": 0})
        if existing and (existing.get("age") and existing.get("gender")):
            return redirect(url_for("user_dashboard", userId=user_id))

        if request.method == "POST":
            name = request.form.get("name", "").strip() or session.get("name")
            phone = request.form.get("phone", "").strip() or session.get("pending_phone")
            age = request.form.get("age")
            gender = request.form.get("gender")

            if not name or not phone or not age or not gender:
                flash("Please fill all fields.", "danger")
            else:
                users.update_one(
                    {"userId": user_id},
                    {"$set": {"userId": user_id, "name": name, "phone": phone, "age": age, "gender": gender, "role": "user"}},
                    upsert=True,
                )
                session["name"] = name
                session["age"] = age
                session["gender"] = gender
                # Clear pending values once saved
                session.pop("pending_name", None)
                session.pop("pending_phone", None)
                flash("Profile saved. Let's start your first test.", "success")
                return redirect(url_for("test", userId=user_id))

        # Pre-fill with whatever we know
        prefill = {
            "name": session.get("name"),
            "phone": session.get("pending_phone"),
            "age": None,
            "gender": None,
        }
        if existing:
            prefill.update({
                "name": existing.get("name") or prefill["name"],
                "phone": existing.get("phone") or prefill["phone"],
                "age": existing.get("age") or prefill["age"],
                "gender": existing.get("gender") or prefill["gender"],
            })

        return render_template("onboarding.html", user=prefill)

    @app.route("/user/<userId>", methods=["GET", "POST"])
    def user_dashboard(userId):
        if not require_login():
            return redirect(url_for("index"))
        if session.get("userId") != userId and not is_admin():
            flash("Unauthorized access to another user's dashboard.", "danger")
            return redirect(url_for("user_dashboard", userId=session.get("userId")))

        user = users.find_one({"userId": userId}, {"_id": 0})
        if request.method == "POST":
            # Save or update basic info
            name = request.form.get("name", "").strip()
            phone = request.form.get("phone", "").strip()
            age = request.form.get("age")
            gender = request.form.get("gender")
            role = user.get("role") if user else "user"
            if not name or not phone:
                flash("Name and phone are required.", "danger")
            else:
                users.update_one(
                    {"userId": userId},
                    {"$set": {"name": name, "phone": phone, "age": age, "gender": gender, "role": role}},
                    upsert=True,
                )
                session["name"] = name
                session["age"] = age
                session["gender"] = gender
                flash("Profile updated.", "success")
            user = users.find_one({"userId": userId}, {"_id": 0})

        if not user:
            # New user flow: create user in database with basic info
            user_data = {
                "userId": userId, 
                "name": session.get("pending_name") or session.get("name"), 
                "phone": session.get("pending_phone"), 
                "age": None, 
                "gender": None, 
                "role": "user"
            }
            users.insert_one(user_data)
            user = user_data

        user_tests = list(tests.find({"userId": userId}, {"_id": 0}).sort("timestamp", -1))

        return render_template("user_dashboard.html", user=user, tests=user_tests)

    @app.route("/test/<userId>", methods=["GET", "POST"])
    def test(userId):
        if not require_login():
            return redirect(url_for("index"))
        if session.get("userId") != userId and not is_admin():
            flash("Unauthorized.", "danger")
            return redirect(url_for("user_dashboard", userId=session.get("userId")))

        user = users.find_one({"userId": userId}, {"_id": 0})
        if not user:
            flash("User not found. Please try logging in again.", "danger")
            return redirect(url_for("index"))
        
        if request.method == "POST":
            form = request.form
            responses = {
                "Smoker": form.get("smoker"),
                "ActivityLevel": form.get("activity"),
                "Diet": form.get("diet"),
                "Alcohol": form.get("alcohol"),
                "SleepHours": float(form.get("sleep", 0) or 0),
                "FamilyHistory": form.get("family"),
                "HighBP": form.get("highbp"),
                "Diabetes": form.get("diabetes"),
                "HeartDisease": form.get("heartdisease"),
                "StressLevel": int(form.get("stress", 0) or 0),
                "Weight": float(form.get("weight", 0) or 0),
                "Height": float(form.get("height", 0) or 0),
                "BMI": float(form.get("bmi", 0) or 0),
                "BloodSugar": float(form.get("bloodsugar", 0) or 0),
                "HeartRate": int(form.get("heartrate", 0) or 0),
                "ECG": form.get("ecg", "")
            }

            # Auto-calc BMI if not provided
            if not responses["BMI"] and responses["Height"] and responses["Weight"]:
                h_m = responses["Height"] / 100.0
                if h_m > 0:
                    responses["BMI"] = round(responses["Weight"] / (h_m * h_m), 2)

            risk = compute_risk_level(responses)

            test_id = str(uuid.uuid4())
            report_meta = {
                "userName": user.get("name") if user else session.get("name"),
                "userId": userId,
                "generatedAt": datetime.utcnow().isoformat() + "Z",
            }
            test_doc = {
                "userId": userId,
                "testId": test_id,
                "responses": responses,
                "riskResult": risk,
                "report": report_meta,
                "timestamp": datetime.utcnow(),
            }
            tests.insert_one(test_doc)

            return redirect(url_for("report", testId=test_id))

        return render_template("test.html", user=user)

    @app.route("/report/<testId>")
    def report(testId):
        if not require_login():
            return redirect(url_for("index"))
        test_doc = tests.find_one({"testId": testId}, {"_id": 0})
        if not test_doc:
            flash("Report not found.", "danger")
            return redirect(url_for("user_dashboard", userId=session.get("userId")))

        # Authorization: allow owner or admin
        if test_doc.get("userId") != session.get("userId") and not is_admin():
            flash("Unauthorized.", "danger")
            return redirect(url_for("user_dashboard", userId=session.get("userId")))

        user = users.find_one({"userId": test_doc["userId"]}, {"_id": 0})
        return render_template("report.html", test=test_doc, user=user)

    @app.route("/report/<testId>/download")
    def download_report(testId):
        if not require_login():
            return redirect(url_for("index"))
        test_doc = tests.find_one({"testId": testId}, {"_id": 0})
        if not test_doc:
            flash("Report not found.", "danger")
            return redirect(url_for("user_dashboard", userId=session.get("userId")))
        if test_doc.get("userId") != session.get("userId") and not is_admin():
            flash("Unauthorized.", "danger")
            return redirect(url_for("user_dashboard", userId=session.get("userId")))

        user = users.find_one({"userId": test_doc["userId"]}, {"_id": 0})
        pdf_path = generate_report_pdf(test_doc, user)
        return send_file(pdf_path, as_attachment=True, download_name=f"CardioClue_Report_{test_doc['testId']}.pdf")

    @app.route("/report/<testId>/share", methods=["POST"])
    def share_report(testId):
        if not require_login():
            return redirect(url_for("index"))
        test_doc = tests.find_one({"testId": testId}, {"_id": 0})
        if not test_doc:
            return jsonify({"ok": False, "message": "Report not found"}), 404
        if test_doc.get("userId") != session.get("userId") and not is_admin():
            return jsonify({"ok": False, "message": "Unauthorized"}), 403

        user = users.find_one({"userId": test_doc["userId"]}, {"_id": 0})
        # Simulate SMS content
        sms_to = request.form.get("phone", user.get("phone") if user else None)
        sms_text = f"CardioClue: Your risk level is {test_doc['riskResult']}. Test ID: {test_doc['testId']}"
        app.logger.info(f"[SIMULATED SMS] To {sms_to}: {sms_text}")

        # Simulate WhatsApp doc by generating PDF
        pdf_path = generate_report_pdf(test_doc, user)
        app.logger.info(f"[SIMULATED WHATSAPP] Sent PDF {pdf_path} to {sms_to}")

        return jsonify({"ok": True, "message": "Shared via SMS and WhatsApp (simulated)."})

    @app.route("/logout")
    def logout():
        session.clear()
        flash("Logged out.", "info")
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


