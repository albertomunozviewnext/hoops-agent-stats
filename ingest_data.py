import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

#1 Cargar variables de entorno
load_dotenv()

def process_csv(file_path, type_label):
    if not os.path.exists(file_path):
        print(f"⚠️ Salteando: {file_path} (no encontrado)")
        return []

    print(f"--- 📊 Analizando {type_label}: {file_path} ---")
    
    # Leemos las primeras 2000 filas para mantener el rendimiento
    df = pd.read_csv(file_path).head(2000)
    # Limpiamos nombres de columnas: minúsculas y sin espacios
    df.columns = df.columns.str.strip().str.lower()

    docs = []
    for _, row in df.iterrows():
        if type_label == "jugador":
            # Columnas típicas de NBA Advanced Player Stats
            player = row.get('player', 'Desconocido')
            team = row.get('team', 'N/A')
            season = row.get('season', 'N/A')
            per = row.get('per', 0)
            ts_pct = row.get('ts_pct', 0)
            usg_pct = row.get('usg_pct', 0)
            bpm = row.get('bpm', 0) # Box Plus/Minus

            content = (f"Jugador: {player} | Temporada: {season} | Equipo: {team} | "
                       f"Eficiencia (PER): {per} | True Shooting: {ts_pct}% | "
                       f"Usage Rate: {usg_pct}% | BPM: {bpm}")
            
            metadata = {"type": "player", "name": player, "season": str(season)}

        else:
            # Columnas típicas de NBA Team Stats (Advanced)
            team = row.get('team', 'Desconocido')
            season = row.get('season', 'N/A')
            off_rtg = row.get('ortg', row.get('off_rating', 0))
            def_rtg = row.get('drtg', row.get('def_rating', 0))
            net_rtg = row.get('nrtg', row.get('net_rating', 0))
            pace = row.get('pace', 0)

            content = (f"Equipo: {team} | Temporada: {season} | "
                       f"Offensive Rating: {off_rtg} | Defensive Rating: {def_rtg} | "
                       f"Net Rating: {net_rtg} | Ritmo (Pace): {pace}")
            
            metadata = {"type": "team", "name": team, "season": str(season)}

        docs.append(Document(page_content=content, metadata=metadata))
    
    return docs

def ingest_nba_stats():

    #Disponemos de dos dataset uno de jugadores y otro de equipos del año 1996 al 2025
    #Estos dataset estan ignorados en github
    csv_path_players = "data/advanced.csv"
    csv_path_teams = "team_advanced.csv"

    all_documents = []