import pickle
import numpy as np  # Explicit import required
from flask import Flask, render_template, request

app = Flask(__name__)

# Safe model loading with error handling
try:
    with open("saved_model.sav", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    model = None

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])  # Fixed: Removed GET
def predict():
    if model is None:
        return render_template("index.html", result="Model not loaded")
    
    try:
        # Convert to numpy array explicitly
        features = np.array([
            float(request.form["SepalLengthCm"]),
            float(request.form["SepalWidthCm"]),
            float(request.form["PetalLengthCm"]),
            float(request.form["PetalWidthCm"])
        ]).reshape(1, -1)  # Ensure correct shape for prediction
        
        result = model.predict(features)[0]
        return render_template("index.html", result=result)
    
    except Exception as e:
        return render_template("index.html", 
                            result=f"Error: {str(e)}",
                            error=True)

if __name__ == "__main__":
    app.run(debug=True)