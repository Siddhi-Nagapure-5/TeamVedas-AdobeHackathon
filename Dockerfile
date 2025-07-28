FROM python:3.10-slim

# System dependencies for OCR + PDF + Image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    gcc \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install Python packages
RUN pip install -r requirements.txt

# Restrict OpenMP threads (for OCR and numpy stability)
ENV OMP_THREAD_LIMIT=4

# Final entrypoint: Process all PDFs
CMD ["python", "process_pdfs.py"]
