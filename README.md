# 🚀 AI Career Pro: RAG-Powered Career Assistant

Este projeto é um assistente de carreira inteligente que utiliza **RAG (Retrieval-Augmented Generation)** para analisar currículos, calcular compatibilidade com vagas de emprego e simular entrevistas técnicas.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.12
- **Interface:** Streamlit (UI/UX customizada)
- **LLM:** Llama 3.3 (via Groq Cloud) para inferência ultra-rápida
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Vector Database:** ChromaDB
- **Orquestração:** LangChain (LCEL)

## ✨ Funcionalidades
- **Chat Consultivo:** Converse com seu currículo e tire dúvidas sobre sua trajetória.
- **Match de Vagas:** Cole a descrição de uma vaga e receba um score de compatibilidade e análise de Gaps.
- **Radar de Skills:** Visualização gráfica das suas competências técnicas.
- **Geradores:** Criação de cartas de apresentação personalizadas e simulados de entrevista.

## 🚀 Como Rodar o Projeto
1. Clone o repositório: `git clone https://github.com/SEU_USUARIO/NOME_REPOSITORIO.git`
2. Instale as dependências: `pip install -r requirements.txt`
3. Configure sua `GROQ_API_KEY` no arquivo `.env` ou no Secrets do Streamlit.
4. Execute: `streamlit run app.py`
