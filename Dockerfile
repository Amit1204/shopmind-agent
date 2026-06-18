FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create output directories
RUN mkdir -p outputs/chroma_db outputs/faiss_index outputs/models

EXPOSE 8000 8501

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
