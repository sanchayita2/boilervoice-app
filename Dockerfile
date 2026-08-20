FROM python:3.11-slim

# Install system dependencies (ffmpeg required for audio handling)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY . .

EXPOSE 10000

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["sh", "-c", "streamlit run app_ui.py --server.port=${PORT:-10000} --server.address=0.0.0.0"]