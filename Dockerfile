# ArticleForge - production image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application (backend + all HTML)
COPY backend.py .
COPY *.html ./

# Expose port
EXPOSE 8000

# Run with uvicorn (bind to 0.0.0.0 for containers)
ENV HOST=0.0.0.0
ENV PORT=8000
CMD uvicorn backend:app --host ${HOST} --port ${PORT}
