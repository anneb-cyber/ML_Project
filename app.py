from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("robot.pkl")  # On réveille le robot

@app.route("/form", methods=["GET", "POST"])
def form():
    html_top = '''
    <html>
    <head>
        <title> Heart disease 💔</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f3f4f6;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                max-width: 400px;
                width: 100%;
            }
            h1 {
                text-align: center;
                color: #4f46e5;
            }
            label {
                display: block;
                margin-top: 10px;
                color: #374151;
            }
            input[type="text"] {
                width: 100%;
                padding: 8px;
                margin-top: 5px;
                border-radius: 5px;
                border: 1px solid #d1d5db;
            }
            input[type="submit"] {
                margin-top: 20px;
                width: 100%;
                background-color: #4f46e5;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .result {
                margin-top: 20px;
                text-align: center;
                font-size: 18px;
                color: #065f46;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>Heart disease 💔</h1>
        <form method="POST">
            <label>age</label>
            <input type="text" name="feature1">

            <label>sex</label>
            <input type="text" name="feature2">

            <label>cp</label>
            <input type="text" name="feature3">

            <label>trestbps</label>
            <input type="text" name="feature4">

            <label>chol</label>
            <input type="text" name="feature5">

            <label>fbs</label>
            <input type="text" name="feature6">

            <label>restecg</label>
            <input type="text" name="feature7">

            <label>thalach</label>
            <input type="text" name="feature8">

            <label>exang</label>
            <input type="text" name="feature9">

            <label>oldpeak</label>
            <input type="text" name="feature10">

            <label>slope</label>
            <input type="text" name="feature11">

            <label>ca</label>
            <input type="text" name="feature12">

            <label>thal</label>
            <input type="text" name="feature13">
            <input type="submit" value="Prédire">
        </form>
    '''

    if request.method == "POST":
        try:
            f1 = float(request.form["feature1"])
            f2 = float(request.form["feature2"])
            f3 = float(request.form["feature3"])
            f4 = float(request.form["feature4"])
            prediction = model.predict([[f1, f2, f3, f4]])
            result = f'<div class="result">Résultat : classe {int(prediction[0])}</div>'
        except:
            result = '<div class="result" style="color:red;">Erreur dans les données saisies !</div>'
        return html_top + result + '</div></body></html>'

    return html_top + '</div></body></html>'



@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    result = model.predict(features)
    return jsonify({"prediction": int(result[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
