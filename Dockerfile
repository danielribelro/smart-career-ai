FROM python:3.11-slim

WORKDIR /app

# Instala apenas o essencial para o Python e Git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos
COPY . .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando de inicialização
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]