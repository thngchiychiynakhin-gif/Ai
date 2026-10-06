from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ==========================================
# Load OrchardGuard AI Model
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "orchardguard_model.pkl"
)

model = joblib.load(MODEL_PATH)

print("==============================================")
print("       OrchardGuard AI API")
print("==============================================")
print("Model loaded successfully")
print("Model:", MODEL_PATH)
print("Soil input range: 0-100%")
print()


# ==========================================
# Health Check
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "OrchardGuard AI API is running",
        "soil_unit": "percent",
        "soil_range": "0-100"
    })


# ==========================================
# AI Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------
        # รับ JSON
        # --------------------------------------

        data = request.get_json()

        if data is None:

            return jsonify({
                "status": "error",
                "message": "Invalid JSON"
            }), 400


        # --------------------------------------
        # ตรวจสอบ Field
        # --------------------------------------

        required_fields = [
            "temperature",
            "humidity",
            "soil",
            "rain"
        ]

        for field in required_fields:

            if field not in data:

                return jsonify({
                    "status": "error",
                    "message": f"Missing field: {field}"
                }), 400


        # --------------------------------------
        # Sensor Data
        # --------------------------------------

        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        soil = float(data["soil"])
        rain = float(data["rain"])


        # --------------------------------------
        # ตรวจสอบ Soil
        # --------------------------------------

        if soil < 0 or soil > 100:

            return jsonify({
                "status": "error",
                "message": "Soil must be between 0 and 100 percent",
                "soil_received": soil
            }), 400


        # --------------------------------------
        # Prepare Features
        # --------------------------------------

        features = np.array([[
            temperature,
            humidity,
            soil,
            rain
        ]])


        # --------------------------------------
        # AI Prediction
        # --------------------------------------

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        classes = model.classes_


        # --------------------------------------
        # Probability Result
        # --------------------------------------

        probability_result = {}

        for class_name, probability in zip(
            classes,
            probabilities
        ):

            probability_result[str(class_name)] = round(
                float(probability) * 100,
                2
            )


        # --------------------------------------
        # ดึง Probability แต่ละระดับ
        # --------------------------------------

        low_probability = probability_result.get(
            "LOW",
            0.0
        )

        medium_probability = probability_result.get(
            "MEDIUM",
            0.0
        )

        high_probability = probability_result.get(
            "HIGH",
            0.0
        )


        # --------------------------------------
        # Risk Score
        # --------------------------------------
        #
        # ใช้ Probability ของระดับที่ AI ทำนาย
        #
        # เช่น
        # LOW    = 25.5
        # MEDIUM = 74.5
        #
        # Prediction = MEDIUM
        # Risk Score = 74.5
        #

        if prediction == "LOW":

            risk_score = low_probability

        elif prediction == "MEDIUM":

            risk_score = medium_probability

        elif prediction == "HIGH":

            risk_score = high_probability

        else:

            risk_score = 0.0


        # --------------------------------------
        # Response
        # --------------------------------------

        result = {

            "status": "success",

            "sensor": {

                "temperature": temperature,

                "humidity": humidity,

                "soil": soil,

                "rain": rain

            },

            "prediction": {

                "risk_level": str(prediction)

            },

            "risk_score": round(
                risk_score,
                2
            ),

            "probability": probability_result

        }


        # --------------------------------------
        # Console
        # --------------------------------------

        print("==============================================")
        print("New Prediction")
        print("Temperature :", temperature)
        print("Humidity    :", humidity)
        print("Soil        :", soil, "%")
        print("Rain        :", rain)
        print("Risk Level  :", prediction)
        print("Risk Score  :", risk_score)
        print("Probability :", probability_result)
        print()


        return jsonify(result)


    # ==========================================
    # Error: Missing Field
    # ==========================================

    except KeyError as e:

        return jsonify({

            "status": "error",

            "message":
                f"Missing field: {e}"

        }), 400


    # ==========================================
    # Error
    # ==========================================

    except Exception as e:

        return jsonify({

            "status": "error",

            "message":
                str(e)

        }), 500


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )