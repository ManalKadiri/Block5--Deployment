# Utiliser une image de base Python
FROM python:3.9-slim

# Installer les dépendances nécessaires
RUN pip install --upgrade pip
RUN pip install streamlit openpyxl matplotlib pandas numpy plotly seaborn

# Copier le code source dans le conteneur
COPY . /home/app

# Définir le répertoire de travail
WORKDIR /home/app

# Exposer le port que Streamlit utilise
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "app.py"]
