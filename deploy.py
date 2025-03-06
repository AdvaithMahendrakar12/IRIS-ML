import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open("saved_model.sav", "rb"))

@app.route("/")
def home():
    return render_template("index.html", result="")  # Initialize result

@app.route("/predict", methods=["POST,GET"])
def predict():
    try:
        sepal_length = float(request.form["SepalLengthCm"])
        sepal_width = float(request.form["SepalWidthCm"])
        petal_length = float(request.form["PetalLengthCm"])
        petal_width = float(request.form["PetalWidthCm"])

        # Predict class
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        return render_template("index.html", result=result)  # Pass result correctly

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")  # Debugging

if __name__ == "__main__":
    app.run(debug=True)