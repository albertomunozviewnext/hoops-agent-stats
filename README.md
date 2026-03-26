# 🏀 Hoops-agent-stats: Advanced Stats & Reasoning

This project is a specialized AI Agent designed to analyze professional basketball through the lens of advanced statistics. Unlike standard RAG systems, this agent uses **Agentic Reasoning** to query deep boxscore data (1997-2025) and provide contextualized insights.

## 🎯 Project Goals (Learning Journey)
This repository marks my transition from basic RAG to **Agentic Engineering**:
1. **Beyond Text:** Implementing RAG on structured data (CSV/Pandas).
2. **Reasoning Loops:** Using **LangGraph** to create a multi-step analysis flow (Thought -> Tool Use -> Observation -> Final Answer).
3. **Gratis Tier Power:** Fully powered by **Google Gemini API** and local vector storage.
4. **Chat-First Architecture:** Replacing static UIs with a dynamic, tool-enabled chatbot.

## 🛠️ Tech Stack
* **Orchestration:** LangChain & LangGraph.
* **Brain:** Google Gemini 1.5 Flash (API).
* **Data Analysis:** Pandas (for advanced boxscore processing).
* **Vector Store:** ChromaDB (for textual context and player profiles).
* **Environment:** Python 3.10+

## 📂 Project Structure
* `/data`: NBA Advanced Boxscores (CSV files).
* `/Docs`: Manuals, glossary of advanced terms (PER, True Shooting, etc.).
* `ingest_data.py`: Script to process historical context and player info.
* `chatbot.py`: The main entry point. An agent-driven CLI to query the data.
* `.env`: API Keys (Ignored by Git).

## 🧠 The "Oracle" Logic
1. **User asks:** "How does 2016 Stephen Curry's efficiency compare to 1998 Michael Jordan?"
2. **The Agent:**
    * **Retrieves** historical context from the vector store.
    * **Executes** a Pandas tool to calculate and filter stats from the CSV.
    * **Reasons** over the differences in pace and league averages.
    * **Responds** with a data-backed scouting report.