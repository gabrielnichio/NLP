from flask import Flask, request, jsonify
from flask_cors import CORS  
import pickle

app = Flask(__name__)
CORS(app)

# Carregar o modelo
with open("language_detector_model.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("le.pkl", "rb") as f:
    le = pickle.load(f)

@app.route("/prever", methods=["POST"])
def prever():
    dados = request.json  # Espera um JSON com os dados de entrada
    phrase = tfidf.transform([dados["entrada"]])
    previsao = modelo.predict(phrase)
    return jsonify({"previsao": le.inverse_transform(previsao)[0]})

if __name__ == "__main__":
    app.run(debug=True)