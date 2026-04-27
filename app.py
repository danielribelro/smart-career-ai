import streamlit as st
import os
import time

# Importações para Nuvem (Groq + HuggingFace)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv() # Carrega a chave do arquivo .env

@st.cache_resource
def load_models():
    # Puxa a chave das 'Secrets' do Streamlit Cloud
    api_key = st.secrets["GROQ_API_KEY"]
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.2,
        groq_api_key=api_key
    )
    
    # HuggingFace roda direto no servidor do Streamlit sem erro
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return llm, embeddings

# 1. CONFIGURAÇÃO DE PÁGINA
st.set_page_config(page_title="AI Career Pro", layout="wide", page_icon="🚀")

# Inicialização de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. DESIGN SYSTEM (CSS AVANÇADO)
def apply_ultra_design():
    st.markdown("""
        <style>
        /* Importando fonte moderna */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at top right, #1a1f25, #0e1117);
        }

        /* Título Estilizado */
        .header-container {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        .main-title {
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .sub-title {
            color: #6c757d;
            font-size: 1.1rem;
        }

        /* Sidebar Customizada */
        [data-testid="stSidebar"] {
            background-color: rgba(22, 27, 34, 0.95);
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        /* Botões Estilo Glassmorphism */
        .stButton>button {
            border-radius: 12px;
            padding: 0.7rem 1.5rem;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.5);
        }

        /* Abas Estilizadas */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #161b22;
            border-radius: 10px;
            color: white;
            padding: 0 25px;
            border: 1px solid #30363d;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(79, 172, 254, 0.1) !important;
            border-color: #4facfe !important;
        }

        /* Chat Bubbles Estilo Moderno */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 16px !important;
            padding: 20px !important;
            margin-bottom: 15px !important;
        }

        /* Inputs e áreas de texto */
        .stTextArea textarea {
            background-color: #0d1117 !important;
            border-radius: 12px !important;
            border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
        }
        
        /* Barra de rolagem */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0e1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

apply_ultra_design()

# 3. BACKEND (MODELOS)
DB_PATH = "./vectorstack"
TMP_DIR = "./temp_pdf"
if not os.path.exists(TMP_DIR): os.makedirs(TMP_DIR)

@st.cache_resource
def load_models():
    llm = ChatOllama(model="llama3", temperature=0.2)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return llm, embeddings

llm, embeddings = load_models()

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

# 4. SIDEBAR REESTILIZADA
with st.sidebar:
    st.markdown("### Centro de Comando")
    uploaded_file = st.file_uploader("Upload do Currículo", type="pdf", label_visibility="collapsed")
    
    if st.button("⚡ ANALISAR AGORA"):
        if uploaded_file:
            with st.spinner("Decodificando arquivo..."):
                file_path = os.path.join(TMP_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(docs)
                
                vectorstore = Chroma.from_documents(
                    documents=splits, embedding=embeddings, persist_directory=DB_PATH
                )
                st.session_state.vectorstore = vectorstore
                
                if not st.session_state.messages:
                    welcome = f"Conexão estabelecida. Analisei seu perfil em '{uploaded_file.name}'. Como posso otimizar sua carreira hoje?"
                    st.session_state.messages.append({"role": "assistant", "content": welcome})
                st.rerun()

# 5. CONTEÚDO PRINCIPAL
st.markdown("""
    <div class="header-container">
        <p class="main-title">AI Career Pro</p>
        <p class="sub-title">Sua trajetória profissional potencializada por inteligência artificial local</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Consultoria de Chat", "🛠️ Laboratório de Documentos"])

with tab1:
    # Container para histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"]=="assistant" else "👤"):
            st.markdown(message["content"])

    if prompt := st.chat_input("Diga algo como: 'Quais cargos combinam comigo?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        if "vectorstore" in st.session_state:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            prompt_template = ChatPromptTemplate.from_template("Contexto: {context}\n\nPergunta: {question}\nResposta sênior:")
            chain = ({"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())

            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                full_response = ""
                result = chain.invoke(prompt)
                for chunk in stream_data(result):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("Aguardando upload de documento na sidebar.")

with tab2:
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Ative o Centro de Comando enviando um PDF primeiro.")
    else:
        st.markdown("### 🎯 Score de Compatibilidade (Match de Vaga)")
        
        # Campo para colar a vaga
        vaga_desc = st.text_area("Cole aqui a descrição da vaga (Requisitos/Tecnologias):", 
                                 placeholder="Ex: Procuramos desenvolvedor Java com experiência em Spring Boot e Docker...",
                                 height=150)

        if st.button("🚀 Calcular Match"):
            if vaga_desc:
                with st.spinner("Analisando compatibilidade..."):
                    # Criando o Retriever para buscar as informações do currículo
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    
                    # Prompt Estruturado para análise técnica
                    match_prompt = ChatPromptTemplate.from_template("""
                        Você é um especialista em recrutamento técnico (Tech Recruiter).
                        Compare o CURRÍCULO fornecido com a VAGA DE EMPREGO abaixo.
                        
                        CURRÍCULO: {context}
                        VAGA: {vaga}
                        
                        Retorne a resposta EXATAMENTE neste formato:
                        SCORE: [Número de 0 a 100]
                        PONTOS FORTES: [Liste 3 pontos]
                        GAPS: [Liste o que falta para o candidato]
                        DICA: [Uma dica de ouro para a entrevista]
                    """)

                    match_chain = (
                        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                         "vaga": RunnablePassthrough()}
                        | match_prompt 
                        | llm 
                        | StrOutputParser()
                    )

                    resultado = match_chain.invoke(vaga_desc)
                    
                    # --- EXIBIÇÃO ESTILIZADA ---
                    st.markdown("---")
                    
                    # Tentar extrair o Score para mostrar um gráfico
                    try:
                        score_val = int([line for line in resultado.split('\n') if 'SCORE:' in line][0].split(':')[1].strip().replace('%',''))
                        st.metric("Índice de Compatibilidade", f"{score_val}%")
                        st.progress(score_val / 100)
                    except:
                        st.info("Análise Concluída:")

                    st.markdown(f"#### 📊 Relatório de Análise\n{resultado}")
            else:
                st.error("Por favor, cole a descrição da vaga para analisar.")

        # Mantendo os botões anteriores abaixo
        st.markdown("---")
        st.markdown("### 🛠️ Outros Geradores")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Gerar Carta de Apresentação"):
                 # (Lógica da carta aqui...)
                 pass
        with c2:
            if st.button("Simular Mock Interview"):
                 # (Lógica da entrevista aqui...)
                 pass