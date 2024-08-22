# Utiliser une image de base Miniconda
FROM continuumio/miniconda3

# Définir le répertoire de travail
WORKDIR /home/app

# Mettre à jour les paquets et installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    nano \
    unzip \
    curl \
    libpq-dev \
    gcc

# Installer le CLI Deta
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Installer AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip && ./aws/install

# Copier le fichier requirements.txt et installer les dépendances Python
COPY requirements.txt /dependencies/requirements.txt
RUN pip install --no-cache-dir -r /dependencies/requirements.txt

# Installer psycopg2-binary pour la connexion PostgreSQL
RUN pip install psycopg2-binary

# Copier un script shell de démarrage
COPY start-mlflow.sh /start-mlflow.sh
RUN chmod +x /start-mlflow.sh

# Exposer le port de l'application
EXPOSE 4000

# Commande pour démarrer le serveur MLflow
CMD ["/start-mlflow.sh"]