import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Téléchargement des ressources NLTK nécessaires
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Chargement du modèle et du vectoriseur
model = joblib.load('client_review_classifier.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Supprimer les caractères spéciaux
    text = re.sub(r'[^\w\s]', '', text)
    
    # Supprimer les URLs
    text = re.sub(r'http\S+', '', text)
    
    # Supprimer les hashtags et les mentions
    text = re.sub(r'#\w+|@\w+', '', text)
    
    # Convertir en minuscules
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Supprimer les stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Fonction pour prédire le sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Interface utilisateur Streamlit
st.title("Analyse des sentiments des avis clients")

user_input = st.text_area("Entrez votre avis client ici:")

if st.button("Analyser le sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        
        if sentiment == 0:
            st.error("Sentiment: Négatif")
        elif sentiment == 1:
            st.success("Sentiment: Positif")
        else:
            st.warning("Sentiment: Neutre")
    else:
        st.warning("Veuillez entrer un avis avant d'analyser.")

st.info("Cette application utilise un modèle d'apprentissage automatique pour analyser le sentiment des avis clients.")
        
        
           # To run the Streamlit app
              #  streamlit run frontend.py      
              
         
  

