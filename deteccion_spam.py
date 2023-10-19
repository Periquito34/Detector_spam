import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv("spam.csv", encoding='Windows-1252')


data = data.rename(columns={'v1': 'etiqueta', 'v2': 'mensaje'})


X = data["mensaje"]
y = data["etiqueta"]
y = y.map({"ham": 0, "spam": 1})


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)


classifier = MultinomialNB()
classifier.fit(X_tfidf, y)

# Solicitar al usuario que ingrese un mensaje
mensaje_usuario = input("Ingresa un mensaje: ")

# Preprocesar el mensaje para que tenga el mismo formato que los datos de entrenamiento
mensaje_usuario_tfidf = tfidf_vectorizer.transform([mensaje_usuario])

# Realizar la predicción
prediccion = classifier.predict(mensaje_usuario_tfidf)

# Imprimir el resultado
if prediccion[0] == 1:
    print("El mensaje es spam.")
else:
    print("El mensaje no es spam.")
