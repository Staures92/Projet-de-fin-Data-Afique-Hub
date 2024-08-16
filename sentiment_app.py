import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')

# Charger le modèle et le vectoriseur
@st.cache_resource
def load_model():
    model = joblib.load('client_review_classifier.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_model()

# Charger les stop words
stop_words = set(stopwords.words('english'))

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
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def analyze_sentiment(review):
    # Prétraitement
    processed_review = preprocess_text(review)
    
    # Vectorisation
    review_vector = vectorizer.transform([processed_review])
    
    # Prédiction
    sentiment = model.predict(review_vector)[0]
    probabilities = model.predict_proba(review_vector)[0]
    
    # Convertir le sentiment en étiquette lisible
    sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    sentiment_label = sentiment_labels[sentiment]
    
    return sentiment_label, probabilities, sentiment_labels

# Interface Streamlit
st.title('Analyse de Sentiment des Avis Clients')

review = st.text_area("Entrez votre avis ici :")

if st.button('Analyser'):
    if review:
        sentiment, probabilities, labels = analyze_sentiment(review)
        
        st.write(f"Sentiment prédit : **{sentiment}**")
        
        st.write("Probabilités :")
        for label, prob in zip(labels, probabilities):
            st.write(f"{label}: {prob:.2f}")
        
        # Visualisation des probabilités
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Bar(x=labels, y=probabilities)])
        fig.update_layout(title='Probabilités des sentiments', xaxis_title='Sentiment', yaxis_title='Probabilité')
        st.plotly_chart(fig)
    else:
        st.write("Veuillez entrer un avis à analyser.")

# Ajouter des informations sur le modèle
st.sidebar.title("À propos du modèle")
st.sidebar.info(
    "Ce modèle utilise un ensemble de classifieurs (VotingClassifier) "
    "comprenant Régression Logistique, Random Forest et SVM. "
    "Il a été entraîné sur un ensemble de données d'avis clients "
    "et utilise le prétraitement de texte et la vectorisation TF-IDF."
)

# Ajouter des exemples d'avis
st.sidebar.title("Exemples d'avis")
st.sidebar.markdown(
    """
    - "This product is amazing! I love it!"
    - "Customer service was horrible. I do not recommend."
    - "It was okay, nothing special."
    - "I had a mixed experience. Some aspects were good, some bad."
    """
)
           # To run the Streamlit app
              #  streamlit run frontend.py      
              
         
  

