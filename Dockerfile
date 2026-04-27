FROM python:3.11-slim

WORKDIR /app

# Instala dependências essenciais do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos do seu repositório
COPY . .

# Instala as bibliotecas do requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando para iniciar o servidor
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]