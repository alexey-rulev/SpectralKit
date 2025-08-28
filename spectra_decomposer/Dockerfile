FROM python:3.11-slim

# Avoid prompts during packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Copy source code
WORKDIR /app
COPY spectra_decomposer ./spectra_decomposer
COPY pyproject.toml .
COPY README.md .

# Install Poetry
RUN pip install --no-cache-dir poetry

# Configure Poetry to install into the system environment (not virtualenv)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi

# Default port used by Streamlit
ENV STREAMLIT_PORT=8501

# Run application
CMD ["sh", "-c", "streamlit run spectra_decomposer/main.py --server.port=$STREAMLIT_PORT"]
