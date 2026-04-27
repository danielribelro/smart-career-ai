import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURAÇÃO DE ALTA PERFORMANCE ---
st.set_page_config(page_title="Career AI | Vision", layout="wide", page_icon="🔮")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. CSS REVOLUCIONÁRIO (GLASSMORPHISM + NEON) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Inter:wght@300;400;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 50% 50%, #12121f 0%, #050508 100%);
        color: #f0f0f0;
    }

    /* Título Futurista */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(2rem, 5vw, 4rem);
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 20px rgba(0, 242, 254, 0.4));
        margin-bottom: 40px;
        letter-spacing: 4px;
    }

    /* Sidebar Estilizada */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Cards de Vidro para Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 30px;
        padding: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        color: #888 !important;
        font-weight: 700;
        transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 20px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #000 !important;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.4);
    }

    /* Botão com Brilho Inteligente */
    div.stButton > button {
        background: linear-gradient(90deg, #7000ff, #4facfe);
        border: none;
        color: white;
        height: 50px;
        border-radius: 15px;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
        transition: 0.5s;
        box-shadow: 0 4px 15px rgba(112, 0, 255, 0.3);
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(112, 0, 255, 0.6);
        border: 1px solid #00f2fe;
    }

    /* Input de Chat Estilo Floating */
    .stChatInputContainer {
        border-radius: 25px !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE: IA ENGINE (OLLAMA) ---
@st.cache_resource
def load_models():
    # Llama 3 via Ollama Local
    llm = Ollama(model="llama3") 
    embeddings = OllamaEmbeddings(model="llama3")
    return llm, embeddings

try:
    llm, embeddings = load_models()
except Exception as e:
    st.error("⚠️ SINAL PERDIDO: O Ollama não responde. Verifique se o daemon está ativo.")
    st.stop()

# --- 4. SIDEBAR: CORE COMMAND CENTER ---
with st.sidebar:
    st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>CORE SYSTEM</h2>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Profile Data (PDF)", type="pdf")
    
    if st.button("Sincronizar IA ⚡"):
        if uploaded_file:
            with st.spinner("Decoding DNA..."):
                if not os.path.exists("temp"): os.makedirs("temp")
                path = os.path.join("temp", uploaded_file.name)
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(path)
                docs = loader.load()
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
                
                st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                
                if not st.session_state.messages:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"🌌 **Nexus Estabelecido.** Perfil de '{uploaded_file.name}' carregado. O que vamos conquistar hoje?"
                    })
                st.success("Sincronia OK")
                st.rerun()

# --- 5. INTERFACE DE COMANDO ---
st.markdown('<p class="main-title">CAREER AI VISION</p>', unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["💬 NEXUS CHAT", "🎯 TARGET MATCH", "🎙️ NEURAL SIM"])

with t1:
    # Renderiza mensagens com estilo futurista
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if p := st.chat_input("Insira sua frequência de comando..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        if "vectorstore" in st.session_state:
            ret = st.session_state.vectorstore.as_retriever()
            template = """Você é uma IA de carreira ultra-avançada. Contexto: {context} | Pergunta: {question}"""
            chain = ({"context": ret | (lambda d: "\n\n".join(x.page_content for x in d)), "question": RunnablePassthrough()} 
                     | ChatPromptTemplate.from_template(template) | llm | StrOutputParser())
            
            with st.chat_message("assistant"):
                res = chain.invoke(p)
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})
        else:
            st.info("💡 Carregue seus dados no Core System para iniciar.")

with t2:
    if "vectorstore" in st.session_state:
        st.markdown("### 🎯 Score de Compatibilidade")
        v = st.text_area("Descreva o alvo (Vaga de Emprego):", height=200)
        if st.button("ANALISAR MATCH"):
            if v:
                with st.spinner("Processando..."):
                    ret = st.session_state.vectorstore.as_retriever()
                    ctx = "\n".join([d.page_content for d in ret.get_relevant_documents(v)])
                    res = llm.invoke(f"Seja um recrutador exigente. Compare: {ctx} com a vaga {v}. Dê uma nota de 0-100 e aponte os erros.")
                    st.write(res)
    else: st.info("🔒 Aguardando entrada de dados.")

with t3:
    if "vectorstore" in st.session_state:
        st.markdown("### 🎙️ Simulação de Entrevista Neural")
        cargo = st.text_input("Qual o seu objetivo?")
        if st.button("GERAR PROTOCOLO"):
            if cargo:
                with st.spinner("Mapeando perguntas..."):
                    ret = st.session_state.vectorstore.as_retriever()
                    ctx = "\n".join([d.page_content for d in ret.get_relevant_documents(cargo)])
                    res = llm.invoke(f"Gere 5 perguntas técnicas de nível SÊNIOR para {cargo} baseadas em: {ctx}")
                    st.markdown("---")
                    st.markdown(res)
            else:
                st.warning("Especifique o cargo.")
    else: st.info("🔒 Sistema offline. Carregue o PDF.")