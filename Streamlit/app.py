

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title='GetAround Project', page_icon='🚗', layout="wide", initial_sidebar_state="auto", menu_items=None)

# Définition de la fonction de la page d'accueil
def main_page():
    DATA_URL = 'get_around_delay_analysis.xlsx'
    DATA_URL2 = 'get_around_pricing_project.csv'

    @st.cache_data
    def load_data():
        data = pd.read_csv(DATA_URL2)
        return data

    @st.cache_data
    def load_data2():
        data2 = pd.read_excel(DATA_URL)
        return data2

    dataset_pricing = load_data()
    df = load_data2()

    st.title("Bienvenue sur notre site webdashboard du projet GET AROUND 🚗")
    st.markdown("")

    # Afficher le dataset - Checkbox
    if st.checkbox('Afficher le dataset'):
        st.write(df)

    st.header('Quelques chiffres..')
    main_metrics_cols = st.columns([33, 33, 34])   # Création de colonnes dans cette partie du webdashboard

    nb_rentals = len(df)
    with main_metrics_cols[0]:
        st.metric(label="Nombre de locations :", value=nb_rentals)
        st.metric(label="Nombre de voitures dans le parc :", value=df['car_id'].nunique())

    with main_metrics_cols[2]:
        st.metric(label="Pourcentage de voitures équipées 'Connect' :", value=f"{round(len(dataset_pricing[dataset_pricing['has_getaround_connect'] == True]) / len(dataset_pricing) * 100)} %")
        st.metric(label="Pourcentage de location via 'Connect' :", value=f"{round(len(df[df['checkin_type'] == 'connect']) / nb_rentals * 100)} %")

    with main_metrics_cols[1]:
        st.metric(label="Pourcentage de locations rendues avec retard :", value=f"{round(len(df[df['delay_at_checkout_in_minutes'] > 0]) / nb_rentals * 100)} %")
        st.metric(label="Pourcentage de locations annulées :", value=f"{round(len(df[df['state'] == 'canceled']) / nb_rentals * 100)} %")

    st.markdown("---")

    st.subheader("Visualisations générales sur le jeu de données")

    # 1ère visualisation
    valeurs_uniques_state = df['state'].value_counts()
    pourcentages_state = valeurs_uniques_state / valeurs_uniques_state.sum() * 100

    fig1, ax1 = plt.subplots()
    ax1.pie(pourcentages_state, labels=pourcentages_state.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Assure que le graphique est bien circulaire
    ax1.set_title('Pourcentage des réservations achevées et annulées')
    st.pyplot(fig1)

    # 2ème visualisation
    valeurs_uniques_checkin_type = df['checkin_type'].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.bar(valeurs_uniques_checkin_type.index, valeurs_uniques_checkin_type.values, color=['skyblue', 'orange'])
    ax2.set_xlabel('checkin_type')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution des modes de check-in')
    ax2.set_xticks(range(len(valeurs_uniques_checkin_type)))
    ax2.set_xticklabels(valeurs_uniques_checkin_type.index, rotation=45)
    st.pyplot(fig2)

    # 3ème visualisation
    df_mobile = df[df['checkin_type'] == 'mobile'].copy()
    df_connect = df[df['checkin_type'] == 'connect'].copy()

    # S'assurer que les colonnes sont de type float pour éviter les erreurs
    df_mobile['delay_at_checkout_in_minutes'] = df_mobile['delay_at_checkout_in_minutes'].astype(float)
    df_connect['delay_at_checkout_in_minutes'] = df_connect['delay_at_checkout_in_minutes'].astype(float)

    # Gérer les valeurs manquantes
    df_mobile = df_mobile.dropna(subset=['delay_at_checkout_in_minutes'])
    df_connect = df_connect.dropna(subset=['delay_at_checkout_in_minutes'])

    conditions_mobile = [
        (df_mobile['delay_at_checkout_in_minutes'] > 0),
        (df_mobile['delay_at_checkout_in_minutes'] == 0),
        (df_mobile['delay_at_checkout_in_minutes'] < 0)
    ]
    choices = ['Retard', 'À temps', 'En avance']

    # Ajouter une valeur par défaut pour les cas non couverts
    df_mobile['checkout_status'] = np.select(conditions_mobile, choices, default='Inconnu')

    conditions_connect = [
        (df_connect['delay_at_checkout_in_minutes'] > 0),
        (df_connect['delay_at_checkout_in_minutes'] == 0),
        (df_connect['delay_at_checkout_in_minutes'] < 0)
    ]

    df_connect['checkout_status'] = np.select(conditions_connect, choices, default='Inconnu')

    df_mobile['checkin_type'] = 'mobile'
    df_connect['checkin_type'] = 'connect'

    df_combined = pd.concat([df_mobile, df_connect])

    status_counts = df_combined.groupby(['checkin_type', 'checkout_status']).size().reset_index(name='Count')

    fig3, ax3 = plt.subplots()
    width = 0.35  # Largeur des barres
    x = np.arange(len(choices))

    # Calcul des valeurs pour chaque type de check-in
    counts_mobile = [status_counts[(status_counts['checkin_type'] == 'mobile') & (status_counts['checkout_status'] == status)]['Count'].sum() for status in choices]
    counts_connect = [status_counts[(status_counts['checkin_type'] == 'connect') & (status_counts['checkout_status'] == status)]['Count'].sum() for status in choices]

    ax3.bar(x - width/2, counts_mobile, width, label='Mobile', color='skyblue')
    ax3.bar(x + width/2, counts_connect, width, label='Connect', color='orange')

    ax3.set_xlabel('Status de Checkout')
    ax3.set_ylabel('Count')
    ax3.set_title('Nombre de checkouts en retard, à temps et en avance par Type de Check-in')
    ax3.set_xticks(x)
    ax3.set_xticklabels(choices, rotation=45)
    ax3.legend(title='Type de Check-in')

    st.pyplot(fig3)

# Définition de la deuxième page
def page2():
    DATA_URL = 'get_around_delay_analysis.xlsx'

    st.title("Dashboard : Analyse du jeu de données fourni par GetAround 🚗💲")
    st.markdown("""
    Voici quelques informations clefs pour comprendre la dynamique des retards lors des réservations sur GetAround 🚗, ainsi que leur impact sur les locations, et donc sur le chiffre d'affaire potentiel de GetAround 🚗.
    """)
    st.markdown("---")

    @st.cache_data
    def load_data2():
        data2 = pd.read_excel(DATA_URL)
        return data2

    df = load_data2()

    # Création de la colonne next_rental_id
    valid_previous_ended_rentals = df.dropna(subset=['previous_ended_rental_id'])

    previous_to_current = valid_previous_ended_rentals.set_index('previous_ended_rental_id')['rental_id'].to_dict()

    df['next_rental_id'] = df['rental_id'].map(previous_to_current)

    # Fonction pour filtrer les outliers
    def filter_outliers(df):
        positive_delays = df[df['delay_at_checkout_in_minutes'] > 0]['delay_at_checkout_in_minutes']

        # Calculer Q1 (25e centile) et Q3 (75e centile)
        Q1 = positive_delays.quantile(0.25)
        Q3 = positive_delays.quantile(0.75)
        IQR = Q3 - Q1

        # Définir les bornes pour les outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Détecter les outliers
        outliers = positive_delays[(positive_delays < lower_bound) | (positive_delays > upper_bound)]

        # Filtrer les lignes sans outliers
        filtered_df = df[
            (df['delay_at_checkout_in_minutes'] > 0) &  # Filtrer les valeurs positives
            (df['delay_at_checkout_in_minutes'] >= lower_bound) &  # Supérieur à la limite inférieure
            (df['delay_at_checkout_in_minutes'] <= upper_bound)  # Inférieur à la limite supérieure
        ]       

        return filtered_df, Q1, Q3, IQR, lower_bound, upper_bound, outliers

    # Filtrer les outliers
    filtered_df, Q1, Q3, IQR, lower_bound, upper_bound, outliers = filter_outliers(df)

    st.subheader("Partie 1 : Impact du délai à instaurer sur le secteur")

    canceled_checkins = df[df['state'] == 'canceled']['checkin_type'].value_counts()

    # Calculer le nombre total de checkins pour chaque mode
    total_checkins = df['checkin_type'].value_counts()

    # Calculer le taux d'annulation
    cancellation_rate = (canceled_checkins / total_checkins) * 100
    cancellation_rate = cancellation_rate.fillna(0)

    # Visualiser avec matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    cancellation_rate.plot(kind='bar', color='skyblue', ax=ax)

    ax.set_title('Taux d\'annulation pour chaque mode de check-in')
    ax.set_xlabel('Mode de check-in')
    ax.set_ylabel('Taux d\'annulation (%)')
    ax.set_xticklabels(cancellation_rate.index, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    # Catégorisation des retards selon le mode de checkin
    # Filtrer les retards > 0
    filtered_df = filtered_df[filtered_df['delay_at_checkout_in_minutes'] > 0]

    # Définir les bins et labels pour les catégories de retard
    bins = [0, 15, 30, 60, 90, 120, float('inf')]
    labels = ['0-15 min', '15-30 min', '30 min-1h', '1h-1h30', '1h30-2h', '>2h']

    # Séparer les données par type de check-in
    filtered_df_mobile = filtered_df[filtered_df['checkin_type'] == 'mobile'].copy()
    filtered_df_connect = filtered_df[filtered_df['checkin_type'] == 'connect'].copy()

    # Ajouter la colonne de catégorie de retard
    filtered_df_mobile['delay_category'] = pd.cut(
        filtered_df_mobile['delay_at_checkout_in_minutes'], bins=bins, labels=labels, right=False
    )
    filtered_df_connect['delay_category'] = pd.cut(
        filtered_df_connect['delay_at_checkout_in_minutes'], bins=bins, labels=labels, right=False
    )

    # Compter le nombre de réservations par catégorie de retard
    delay_category_counts_mobile = filtered_df_mobile['delay_category'].value_counts().sort_index()
    delay_category_counts_connect = filtered_df_connect['delay_category'].value_counts().sort_index()

    # Visualisation avec Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # Largeur des barres
    x = np.arange(len(labels))  # Positions des barres

    ax.bar(x - width/2, delay_category_counts_mobile.values, width, label='Mobile', color='skyblue')
    ax.bar(x + width/2, delay_category_counts_connect.values, width, label='Connect', color='orange')

    ax.set_xlabel('Catégorie de Retard')
    ax.set_ylabel('Nombre de Réservations')
    ax.set_title('Distribution des Retards de Checkout par Mode de Check-in')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(title='Mode de Check-in')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Afficher le graphique avec Streamlit
    st.pyplot(fig)
    st.subheader("Partie 2 : Nombre de locations qui seront potentiellement impactées")

    impacted_df = filtered_df.dropna(subset=['time_delta_with_previous_rental_in_minutes']).copy()
    impacted_df.loc[:, 'difference'] = impacted_df['time_delta_with_previous_rental_in_minutes'] - impacted_df['delay_at_checkout_in_minutes']
    # Calcul des impacts et des cas résolus

    threshold_range = np.arange(0, 60*12, step=15) # 15min intervals for 12 hours
    impacted_list_mobile = []
    impacted_list_connect = []
    impacted_list_total = []
    solved_list_mobile = []
    solved_list_connect = []
    solved_list_total = []

    solved_list = []
    for t in threshold_range:
        impacted = impacted_df.dropna(subset=['time_delta_with_previous_rental_in_minutes'])
        connect_impact = impacted[impacted['checkin_type'] == 'connect']
        mobile_impact = impacted[impacted['checkin_type'] == 'mobile']
        connect_impact = connect_impact[connect_impact['time_delta_with_previous_rental_in_minutes'] < t]
        mobile_impact = mobile_impact[mobile_impact['time_delta_with_previous_rental_in_minutes'] < t]
        impacted = impacted[impacted['time_delta_with_previous_rental_in_minutes'] < t]
        impacted_list_connect.append(len(connect_impact))
        impacted_list_mobile.append(len(mobile_impact))
        impacted_list_total.append(len(impacted))

        solved = impacted_df[impacted_df['difference'] < 0]
        connect_solved = solved[solved['checkin_type'] == 'connect']
        mobile_solved = solved[solved['checkin_type'] == 'mobile']
        connect_solved = connect_solved[connect_solved['delay_at_checkout_in_minutes'] < t]
        mobile_solved = mobile_solved[mobile_solved['delay_at_checkout_in_minutes'] < t]
        solved = solved[solved['delay_at_checkout_in_minutes'] < t]
        solved_list_connect.append(len(connect_solved))
        solved_list_mobile.append(len(mobile_solved))
        solved_list_total.append(len(solved))

    # Créer le graphique avec Matplotlib pour les impacts
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_range, impacted_list_connect, label='Connect impacted', marker='o', linestyle='-')
    ax.plot(threshold_range, impacted_list_mobile, label='Mobile impacted', marker='o', linestyle='-')
    ax.plot(threshold_range, impacted_list_total, label='Total impacted', marker='o', linestyle='-')

    ax.set_xlabel('Threshold (min)')
    ax.set_ylabel('Number of impacted cases')
    ax.set_title('Number of Impacted Cases by Threshold')
    ax.grid(True)
    ax.legend()

    # Afficher le graphique avec Streamlit
    st.pyplot(fig)

    # Fonction pour calculer l'impact
    def calculate_impact(threshold):
        impacted_df = filtered_df.dropna(subset=['time_delta_with_previous_rental_in_minutes']).copy()
        impacted_df['difference'] = impacted_df['time_delta_with_previous_rental_in_minutes'] - impacted_df['delay_at_checkout_in_minutes']

        impacted = impacted_df[impacted_df['time_delta_with_previous_rental_in_minutes'] < threshold]
        connect_impact = impacted[impacted['checkin_type'] == 'connect']
        mobile_impact = impacted[impacted['checkin_type'] == 'mobile']

        #solved = impacted_df[impacted_df['difference'] < 0]
        #connect_solved = solved[solved['checkin_type'] == 'connect']
        #mobile_solved = solved[solved['checkin_type'] == 'mobile']
        #connect_solved = connect_solved[connect_solved['delay_at_checkout_in_minutes'] < threshold]
        #mobile_solved = mobile_solved[mobile_solved['delay_at_checkout_in_minutes'] < threshold]

        return {
            'impacted_connect': len(connect_impact),
            'impacted_mobile': len(mobile_impact),
            'impacted_total': len(impacted),
            #'solved_connect': len(connect_solved),
            #'solved_mobile': len(mobile_solved),
            #'solved_total': len(solved)
        }

    # Curseur pour sélectionner le délai
    threshold = st.slider('Choisissez le délai (en minutes) entre deux locations', 0, 60*12, 45, step=15)

    # Calculer l'impact pour le délai sélectionné
    results = calculate_impact(threshold)

    # Afficher les résultats
    st.write(f"**Nombre de locations impactées pour un délai de {threshold} minutes :**")
    st.write(f"Check-ins connectés : {results['impacted_connect']}")
    st.write(f"Check-ins mobiles : {results['impacted_mobile']}")
    st.write(f"Total : {results['impacted_total']}")

    #st.write(f"**Nombre de locations résolues pour un délai de {threshold} minutes :**")
    #st.write(f"Check-ins connectés : {results['solved_connect']}")
    #st.write(f"Check-ins mobiles : {results['solved_mobile']}")
    #st.write(f"Total : {results['solved_total']}")

    # Calculer les impacts et résolutions pour une gamme de seuils
    #threshold_range = np.arange(0, 60*12, step=15)
    #impacted_list_connect = []
    #impacted_list_mobile = []
    #impacted_list_total = []
    #solved_list_connect = []
    #solved_list_mobile = []
    #solved_list_total = []

    #for t in threshold_range:
        #res = calculate_impact(t)
        #impacted_list_connect.append(res['impacted_connect'])
        #impacted_list_mobile.append(res['impacted_mobile'])
        #impacted_list_total.append(res['impacted_total'])
        #solved_list_connect.append(res['solved_connect'])
        #solved_list_mobile.append(res['solved_mobile'])
        #solved_list_total.append(res['solved_total'])

    # Créer les graphiques avec Matplotlib
    #fig1, ax1 = plt.subplots(figsize=(10, 6))
    #ax1.plot(threshold_range, solved_list_connect, label='Connect solved', marker='o', color='blue')
    #ax1.plot(threshold_range, solved_list_mobile, label='Mobile solved', marker='o', color='green')
    #ax1.plot(threshold_range, solved_list_total, label='Total solved', marker='o', color='red')
    #ax1.set_title('Cas résolus en fonction du seuil de délai')
    #ax1.set_xlabel('Seuil (min)')
    #ax1.set_ylabel('Nombre de cas résolus')
    #ax1.grid(True)
    #ax1.legend()

    #fig2, ax2 = plt.subplots(figsize=(10, 6))
    #ax2.plot(threshold_range, impacted_list_connect, label='Connect impacted', marker='o', color='blue')
    #ax2.plot(threshold_range, impacted_list_mobile, label='Mobile impacted', marker='o', color='green')
    #ax2.plot(threshold_range, impacted_list_total, label='Total impacted', marker='o', color='red')
    #ax2.set_title('Cas impactés en fonction du seuil de délai')
    #ax2.set_xlabel('Seuil (min)')
    #ax2.set_ylabel('Nombre de cas impactés')
    #ax2.grid(True)
    #ax2.legend()

    # Afficher les graphiques dans Streamlit
    #st.pyplot(fig1)
    #st.pyplot(fig2)

    st.subheader("Partie 3 : Occurrences des retards de checkout et impact sur le conducteur suivant")

    # Occurrence des retards pour les prochains checkout

    df_mobile = filtered_df[filtered_df['checkin_type'] == 'mobile'].copy()
    df_connect = filtered_df[filtered_df['checkin_type'] == 'connect'].copy()

    df_mobile = df_mobile.dropna(subset=['delay_at_checkout_in_minutes'])
    df_connect = df_connect.dropna(subset=['delay_at_checkout_in_minutes'])

    df_mobile = df_mobile[df_mobile['delay_at_checkout_in_minutes'] > 0]
    df_connect = df_connect[df_connect['delay_at_checkout_in_minutes'] > 0]

    df_mobile['checkout_status'] = 'Retard'
    df_connect['checkout_status'] = 'Retard'

    df_combined = pd.concat([df_mobile, df_connect])

    status_counts = df_combined.groupby(['checkin_type', 'checkout_status']).size().reset_index(name='Count')


    # Créer le graphique Plotly
    fig = px.bar(
        status_counts, 
        x='checkout_status', 
        y='Count', 
        color='checkin_type',
        barmode='group',
        labels={'checkout_status': 'Status de Checkout', 'Count': 'Count', 'checkin_type': 'Type de Check-in'},
        title='Nombre de checkouts en retard par Type de Check-in',
        width=800, 
        height=600
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)


    # Calcul de l'impact du retard des locations sur les suivantes

    filtered_df_clean = filtered_df.dropna(subset=['next_rental_id'])
    filtered_df_clean.reset_index(drop=True, inplace=True)

    # Créer un dictionnaire pour mapper rental_id à delay_at_checkout_in_minutes
    delay_dict = filtered_df_clean.set_index('rental_id')['delay_at_checkout_in_minutes'].to_dict()

    # Ajouter la colonne 'next_delay' en utilisant le dictionnaire, en vérifiant l'existence des clés
    filtered_df_clean['next_delay'] = filtered_df_clean['next_rental_id'].map(delay_dict).fillna(0).astype(int)

    # Calculer l'impact des retards 
    filtered_df_clean['impact_on_next_rental'] = filtered_df_clean['next_delay'] - filtered_df_clean['delay_at_checkout_in_minutes'].fillna(0).astype(int)

    # Convertir les colonnes en entier
    filtered_df_clean['delay_at_checkout_in_minutes'] = filtered_df_clean['delay_at_checkout_in_minutes'].fillna(0).astype(int)
    filtered_df_clean['impact_on_next_rental'] = filtered_df_clean['impact_on_next_rental'].astype(int)
    filtered_df_clean['previous_ended_rental_id'] = filtered_df_clean['previous_ended_rental_id'].fillna(0).astype(int)
    filtered_df_clean['next_rental_id'] = filtered_df_clean['next_rental_id'].fillna(0).astype(int)

    filtered_df_clean= filtered_df_clean.loc[filtered_df_clean['next_delay'] > 0]

    # Visualisation de l'impact des retards de checkout sur les locations suivantes
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=filtered_df_clean, 
        x='delay_at_checkout_in_minutes', 
        y='next_delay', 
        hue='checkin_type', 
        style='checkin_type', 
        palette='deep', 
        ax=ax
    )

    sns.regplot(
        data=filtered_df_clean, 
        x='delay_at_checkout_in_minutes', 
        y='next_delay', 
        scatter=False, 
        color='blue', 
        ax=ax
    )

    ax.set_title('Impact des retards des locations sur les suivantes')
    ax.set_xlabel('Retard au checkout en minutes')
    ax.set_ylabel('Impact sur la location suivante (en minutes)')
    ax.legend(title='Type de check-in')
    ax.grid(True)

    # Afficher le graphique avec Streamlit
    st.pyplot(fig)

    st.subheader("Partie 4 : Nombres de cas qui seront potentiellement résolus")

    # Fonction pour calculer les cas résolus
    def calculate_solved(threshold):
        impacted_df = filtered_df.dropna(subset=['time_delta_with_previous_rental_in_minutes']).copy()
        impacted_df['difference'] = impacted_df['time_delta_with_previous_rental_in_minutes'] - impacted_df['delay_at_checkout_in_minutes']

        # Trouver les cas résolus
        solved = impacted_df[impacted_df['difference'] < 0]
        connect_solved = solved[solved['checkin_type'] == 'connect']
        mobile_solved = solved[solved['checkin_type'] == 'mobile']
        connect_solved = connect_solved[connect_solved['delay_at_checkout_in_minutes'] < threshold]
        mobile_solved = mobile_solved[mobile_solved['delay_at_checkout_in_minutes'] < threshold]

        return {
            'solved_connect': len(connect_solved),
            'solved_mobile': len(mobile_solved),
            'solved_total': len(connect_solved) + len(mobile_solved),
        }

    # Curseur pour sélectionner le délai (ajout d'une clé unique)
    threshold = st.slider('Choisissez le délai (en minutes) entre deux locations', 0, 60*12, 45, step=15, key='threshold_slider')

    # Calculer les cas résolus pour le seuil sélectionné
    results = calculate_solved(threshold)

    # Afficher les résultats
    st.write(f"**Nombre de locations résolues pour un délai de {threshold} minutes :**")
    st.write(f"Check-ins connectés : {results['solved_connect']}")
    st.write(f"Check-ins mobiles : {results['solved_mobile']}")
    st.write(f"Total : {results['solved_total']}")

    # Calculer les cas résolus pour chaque seuil dans la gamme
    threshold_range = np.arange(0, 60*12, step=15)
    solved_list_connect = []
    solved_list_mobile = []
    solved_list_total = []

    for t in threshold_range:
        res = calculate_solved(t)
        solved_list_connect.append(res['solved_connect'])
        solved_list_mobile.append(res['solved_mobile'])
        solved_list_total.append(res['solved_total'])

    # Créer le graphique avec Matplotlib pour les cas résolus
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_range, solved_list_connect, label='Connect solved', marker='o', linestyle='-')
    ax.plot(threshold_range, solved_list_mobile, label='Mobile solved', marker='o', linestyle='-')
    ax.plot(threshold_range, solved_list_total, label='Total solved', marker='o', linestyle='-')

    # Mettre en surbrillance le point correspondant au seuil sélectionné
    ax.scatter(threshold, results['solved_connect'], color='blue', s=100, zorder=5)
    ax.scatter(threshold, results['solved_mobile'], color='orange', s=100, zorder=5)
    ax.scatter(threshold, results['solved_total'], color='green', s=100, zorder=5)

    ax.set_xlabel('Seuil (min)')
    ax.set_ylabel('Nombre de cas résolus')
    ax.set_title('Cas résolus en fonction du seuil de délai')
    ax.grid(True)
    ax.legend()

    # Afficher le graphique avec Streamlit
    st.pyplot(fig)



# Choix de la page à afficher
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse des retards"])

if page == "Accueil":
    main_page()
else:
    page2()
