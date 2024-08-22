# Utilisez l'image Python 3.9 comme base
FROM python:3.9

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copiez le fichier api.py dans le conteneur
COPY api.py /app/

# Exposez le port que l'application utilisera
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-5000}"]

