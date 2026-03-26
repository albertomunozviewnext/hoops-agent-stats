import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Cargar configuración
load_dotenv()

def get_chatbot_response(user_query):
    # 2. Configurar el modelo y los embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 3. Cargar la base de datos que creamos antes
    vectorstore = Chroma(
        persist_directory="./db_nba_stats", 
        embedding_function=embeddings
    )
    
    # Configuramos el buscador (retriever)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. Definir el "System Prompt" (La personalidad del bot)
    system_prompt = (
        "Eres un analista experto de la NBA. Usa los siguientes fragmentos de "
        "contexto recuperado para responder la pregunta del usuario."
        "Si no sabes la respuesta basándote en el contexto, di que no tienes "
        "esos datos específicos, pero no inventes estadísticas."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 5. Crear la cadena de RAG (Retrieval-Augmented Generation)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 6. Ejecutar y devolver respuesta
    response = rag_chain.invoke({"input": user_query})
    return response["answer"]

# --- Bloque para probarlo en la terminal ---
if __name__ == "__main__":
    print("🏀 NBA Agent está listo. Escribe 'salir' para terminar.")
    while True:
        query = input("\nPregunta: ")
        if query.lower() in ["salir", "exit", "quit"]:
            break
        
        print("🤔 Pensando...")
        try:
            res = get_chatbot_response(query)
            print(f"\n🤖 Bot: {res}")
        except Exception as e:
            print(f"❌ Error: {e}")