FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose default port
EXPOSE 8477

# Default command: run server
CMD ["python", "-m", "imprint", "serve", "8477"]
