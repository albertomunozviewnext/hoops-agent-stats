import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

def process_csv(file_path, type_label):
    if not os.path.exists(file_path):
        print(f"⚠️ Archivo no encontrado: {file_path}")
        return []

    print(f"--- 📊 Procesando {type_label}: {file_path} ---")
    
    # Cargamos el CSV
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    # FILTRO POR FECHA (Usando la columna 'date')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Nos quedamos solo con datos de 2024 y 2025 para no saturar
        df = df[df['date'] >= '2024-01-01']
    
    # Si aún así hay muchos, limitamos a 150 filas para asegurar el éxito
    df = df.head(150)

    docs = []
    for _, row in df.iterrows():
        fecha_str = row['date'].strftime('%Y-%m-%d') if 'date' in row else "N/A"
        
        if type_label == "jugador":
            content = (f"Fecha: {fecha_str} | Jugador: {row.get('player', 'N/A')} | "
                       f"Equipo: {row.get('team', 'N/A')} | PER: {row.get('per', 0)}")
            metadata = {"type": "player", "date": fecha_str}
        else:
            content = (f"Fecha: {fecha_str} | Equipo: {row.get('team', 'N/A')} | "
                       f"Ortg: {row.get('ortg', 0)} | Drtg: {row.get('drtg', 0)}")
            metadata = {"type": "team", "date": fecha_str}
        
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def ingest_all_data():
    # 1. Rutas de tus archivos
    path_players = "data/advanced.csv" 
    path_teams = "data/team_advanced.csv"

    # 2. Recolectar todos los documentos
    all_docs = []
    all_docs.extend(process_csv(path_players, "jugador"))
    all_docs.extend(process_csv(path_teams, "equipo"))

    if not all_docs:
        print("❌ No hay datos para procesar.")
        return

    # 3. Configurar Embeddings (Modelo correcto y estable)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Ingesta por Lotes (Para evitar el error 429)
    print(f"--- 🧠 Iniciando ingesta de {len(all_docs)} documentos en lotes... ---")
    
    # Creamos la base de datos vacía con el primer lote
    batch_size = 50
    vectorstore = Chroma(persist_directory="./db_nba_stats", embedding_function=embeddings)

    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"✅ Lote {i//batch_size + 1} completado ({min(i + batch_size, len(all_docs))}/{len(all_docs)})")
        # Pausa de seguridad para que la API respire
        time.sleep(2)

    print("\n🔥 ¡TODO LISTO! Base de datos creada en ./db_nba_stats")

if __name__ == "__main__":
    ingest_all_data()


     
    