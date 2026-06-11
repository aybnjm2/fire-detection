import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import os
import joblib
from tensorflow.keras.models import load_model

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, SCALER_PATH, CLUSTERER_PATH, REG_MODEL_PATH, CLS_MODEL_PATH
from src.inference import predict_new_anomaly
from src.agent import FireMonitoringAgent, FIRE_TYPE_LABELS, ALERT_LOG_PATH


# CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Surveillance des Feux de Forêt", layout="wide")

# Style CSS minimaliste pour nettoyer l'interface
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-weight: 400; }
    </style>
""", unsafe_allow_html=True)

st.title("Surveillance des Feux de Forêt")


# GESTION D'ÉTAT (SESSION STATE)
# Permet de ne pas perdre les résultats lors de la navigation entre les onglets
if 'sim_result' not in st.session_state:
    st.session_state.sim_result = None
if 'sim_conditions' not in st.session_state:
    st.session_state.sim_conditions = None
if 'sim_niveau' not in st.session_state:
    st.session_state.sim_niveau = None
if 'agent_alerts' not in st.session_state:
    st.session_state.agent_alerts = None


# CHARGEMENT DES DONNÉES
@st.cache_data
def load_historical_data():
    mock_file = os.path.join(DATA_RAW_DIR, 'MODIS_fires.csv')
    if os.path.exists(mock_file):
        df = pd.read_csv(mock_file)
        if 'cluster_id' not in df.columns:
            df['cluster_id'] = np.random.choice([-1, 0, 1, 2], size=len(df))
        if 'acq_date' in df.columns:
            df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'latitude':   np.random.uniform(36.0, 39.0, 500),
            'longitude':  np.random.uniform(35.0, 42.0, 500),
            'brightness': np.random.uniform(300, 350, 500),
            'frp':        np.random.uniform(5, 120, 500),
            'confidence': np.random.randint(40, 100, 500),
            'daynight':   np.random.choice(['D', 'N'], 500),
            'satellite':  np.random.choice(['Terra', 'Aqua'], 500),
            'type':       np.random.choice([0, 1, 2, 3], 500),
            'cluster_id': np.random.choice([-1, 0, 1, 2, 3], 500, p=[0.4, 0.2, 0.15, 0.15, 0.1])
        })
        df['acq_date'] = pd.date_range(start='2022-01-01', periods=500, freq='6h')
    return df

df = load_historical_data()
models_exist = all(os.path.exists(p) for p in [SCALER_PATH, CLUSTERER_PATH, REG_MODEL_PATH, CLS_MODEL_PATH])


# NAVIGATION
tab_carte, tab_sim, tab_rapport, tab_alerte = st.tabs([
    "Carte Historique", 
    "Simulation", 
    "Rapport", 
    "Surveillance Active"
])


# ONGLET 1 : Carte Historique
with tab_carte:
    st.header("Carte des détections")
    
    filtre_periode = st.radio("Période de détection", options=["Toutes", "Jour", "Nuit"], horizontal=True)

    filtered_df = df.copy()
    if filtre_periode != "Toutes":
        val = 'D' if filtre_periode == 'Jour' else 'N'
        filtered_df = filtered_df[filtered_df['daynight'] == val]

    # Simplification des couleurs : Gris pour isolé, Rouge pour foyers groupés
    def get_simple_color(cluster_id):
        if cluster_id == -1: return [150, 150, 150, 150]
        return [220, 50, 50, 200]

    filtered_df['color'] = filtered_df['cluster_id'].apply(get_simple_color)

    view_state = pdk.ViewState(
        latitude=filtered_df['latitude'].mean() if not filtered_df.empty else 37.0,
        longitude=filtered_df['longitude'].mean() if not filtered_df.empty else 38.0,
        zoom=5, pitch=0
    )
    layer = pdk.Layer(
        "ScatterplotLayer", data=filtered_df,
        get_position='[longitude, latitude]',
        get_color='color', get_radius=6000, pickable=True
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer], initial_view_state=view_state,
        tooltip={"text": "Emplacement de l'anomalie"}
    ))


# ONGLET 2 : Simulation
with tab_sim:
    st.header("Simulation de détection")

    if not models_exist:
        st.warning("Le système n'est pas entièrement initialisé. Modèles d'analyse introuvables.")
    else:
        niveau_risque = st.selectbox("Sélectionnez un niveau de risque à simuler", ["Faible", "Moyen", "Élevé"])
        
        if st.button("Lancer la simulation"):
            with st.spinner("Génération et analyse en cours..."):
                # Génération des valeurs environnementales réalistes selon le risque
                if niveau_risque == "Faible":
                    temp = np.random.uniform(15, 22)
                    hum = np.random.uniform(50, 70)
                    vent = np.random.uniform(5, 15)
                    secheresse = "Faible"
                    puissance = np.random.uniform(5, 15)
                    modis_bright = np.random.uniform(300, 310)
                    modis_t31 = np.random.uniform(290, 295)
                    modis_conf = 50
                elif niveau_risque == "Moyen":
                    temp = np.random.uniform(23, 32)
                    hum = np.random.uniform(30, 49)
                    vent = np.random.uniform(16, 30)
                    secheresse = "Modérée"
                    puissance = np.random.uniform(16, 45)
                    modis_bright = np.random.uniform(315, 330)
                    modis_t31 = np.random.uniform(295, 305)
                    modis_conf = 75
                else: # Élevé
                    temp = np.random.uniform(33, 45)
                    hum = np.random.uniform(10, 29)
                    vent = np.random.uniform(31, 60)
                    secheresse = "Forte"
                    puissance = np.random.uniform(46, 120)
                    modis_bright = np.random.uniform(340, 370)
                    modis_t31 = np.random.uniform(305, 320)
                    modis_conf = 95

                # Sauvegarde dans la session
                st.session_state.sim_conditions = {
                    "Température": temp, "Humidité": hum, "Vent": vent, 
                    "Sécheresse": secheresse, "Puissance": puissance
                }
                st.session_state.sim_niveau = niveau_risque

                # Analyse système
                new_data = {
                    'latitude': 36.75, 'longitude': 36.22,
                    'brightness': modis_bright, 'scan': 1.2, 'track': 1.1,
                    'confidence': modis_conf, 'bright_t31': modis_t31,
                    'daynight': 'D', 'satellite': 'Terra'
                }
                
                try:
                    st.session_state.sim_result = predict_new_anomaly(new_data)
                except Exception:
                    st.error("Erreur technique lors de l'exécution du modèle d'analyse.")
                    st.session_state.sim_result = None

        # Affichage des résultats sauvegardés dans l'état de session
        if st.session_state.sim_result and st.session_state.sim_conditions:
            cond = st.session_state.sim_conditions
            
            st.subheader("Conditions environnementales générées")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Température", f"{cond['Température']:.1f} °C")
            col2.metric("Humidité", f"{cond['Humidité']:.1f} %")
            col3.metric("Vent", f"{cond['Vent']:.1f} km/h")
            col4.metric("Sécheresse", cond['Sécheresse'])
            col5.metric("Puissance thermique", f"{cond['Puissance']:.1f} MW")

            st.divider()

            result = st.session_state.sim_result
            fire_types = {0: "Géothermique", 1: "Industriel", 2: "Végétation", 3: "Autre"}
            type_feu = fire_types.get(result.get('Predicted_Fire_Type', -1), "Inconnu")
            
            st.subheader("Diagnostic du système")
            res1, res2 = st.columns(2)
            res1.metric("Type d'événement identifié", type_feu)
            res2.metric("Niveau de risque évalué", st.session_state.sim_niveau.upper())


# ONGLET 3 : Rapport (Business Intelligence)
with tab_rapport:
    st.header("Rapport global d'activité")

    agent_log = FireMonitoringAgent.load_alert_log()
    
    if not agent_log.empty:
        bi_df = agent_log.copy()
        bi_df['type_label'] = bi_df['fire_type']
        bi_df['severity'] = bi_df['severity'].replace({
            "LOW": "Faible", "MEDIUM": "Moyen", "HIGH": "Élevé", "CRITICAL": "Critique"
        })
    else:
        bi_df = df.copy()
        bi_df['type_label'] = bi_df['type'].map({
            0: "Géothermique", 1: "Industriel", 2: "Végétation", 3: "Autre"
        }).fillna("Inconnu")
        bi_df['severity'] = pd.cut(
            bi_df.get('frp', pd.Series(np.random.uniform(5, 120, len(bi_df)))),
            bins=[-np.inf, 30, 60, 100, np.inf],
            labels=["Faible", "Moyen", "Élevé", "Critique"]
        )

    # Indicateurs globaux
    total_events = len(bi_df)
    critical_count = (bi_df['severity'] == 'Critique').sum()
    veg_count = (bi_df['type_label'].str.contains("Végétation|Vegetation", case=False, na=False)).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Détections totales", f"{total_events:,}")
    col2.metric("Situations critiques", f"{critical_count}")
    col3.metric("Feux de végétation", f"{veg_count}")

    st.divider()

    # Graphiques épurés
    gcol1, gcol2 = st.columns(2)
    
    with gcol1:
        type_counts = bi_df['type_label'].value_counts().reset_index()
        type_counts.columns = ['type', 'count']
        fig_pie = px.pie(type_counts, names='type', values='count', title="Répartition par type d'événement", hole=0.5)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with gcol2:
        sev_counts = bi_df['severity'].value_counts().reset_index()
        sev_counts.columns = ['severity', 'count']
        fig_bar = px.bar(sev_counts, x='severity', y='count', title="Niveau de gravité des détections")
        fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis_title="Gravité", yaxis_title="Nombre")
        fig_bar.update_traces(marker_color='#4051b5')
        st.plotly_chart(fig_bar, use_container_width=True)


# ONGLET 4 : Surveillance Active (Agent IA)
with tab_alerte:
    st.header("Surveillance continue")
    
    if not models_exist:
        st.warning("Veuillez patienter que le système principal soit opérationnel avant de lancer la surveillance.")
    else:
        if st.button("Vérifier la situation actuelle"):
            with st.spinner("Analyse des dernières données satellitaires en cours..."):
                try:
                    # Nouveau code : Active Discord et le mode test
                    agent = FireMonitoringAgent(
                        enable_discord=True,         # Active l'envoi vers Discord
                        generate_test_critical=True  # Mettez False une fois le test réussi !
                    )
                    st.session_state.agent_alerts = agent.run_cycle()
                except Exception:
                    st.error("Impossible de récupérer ou d'analyser les nouvelles données pour le moment.")
                    st.session_state.agent_alerts = None

        # Affichage des alertes sauvegardées dans l'état de session
        if st.session_state.agent_alerts is not None:
            alerts = st.session_state.agent_alerts
            if alerts:
                st.success("Analyse terminée.")
                for alert in alerts:
                    st.subheader(f"Détection : {alert.fire_type}")
                    c1, c2, c3 = st.columns(3)
                    
                    s_traduction = {"LOW": "Faible", "MEDIUM": "Moyen", "HIGH": "Élevé", "CRITICAL": "Critique"}
                    niveau = s_traduction.get(alert.severity, "Inconnu")
                    
                    c1.write(f"**Localisation:** {alert.latitude:.3f}, {alert.longitude:.3f}")
                    c2.write(f"**Gravité:** {niveau}")
                    c3.write(f"**Puissance thermique:** {alert.predicted_frp_mw:.1f} MW")
                    
                    st.write(f"**Explication:** {alert.reasoning}")
                    if alert.escalated:
                        st.error("Action requise : Situation escaladée aux équipes de réponse.")
                    st.divider()
            else:
                st.info("Aucune nouvelle observation nécessitant une attention immédiate.")

    st.subheader("Historique récent")
    log_df = FireMonitoringAgent.load_alert_log()

    if log_df.empty:
        st.info("Aucun historique disponible.")
    else:
        # Simplification des données affichées pour la lisibilité
        log_df['severity'] = log_df['severity'].replace({
            "LOW": "Faible", "MEDIUM": "Moyen", "HIGH": "Élevé", "CRITICAL": "Critique"
        })
        log_df['escalated'] = log_df['escalated'].map({True: "Oui", False: "Non"})
        
        colonnes_utiles = {
            'timestamp': 'Date', 
            'fire_type': 'Type', 
            'severity': 'Gravité', 
            'predicted_frp_mw': 'Puissance (MW)',
            'escalated': 'Action requise'
        }
        
        display_df = log_df[[c for c in colonnes_utiles.keys() if c in log_df.columns]].rename(columns=colonnes_utiles)
        st.dataframe(display_df, use_container_width=True, hide_index=True)