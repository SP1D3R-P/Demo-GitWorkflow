FROM python:3.11-slim-bookworm

WORKDIR /app
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434/
COPY pyproject.toml uv.lock ./
RUN pip install uv
RUN uv sync

COPY . .

EXPOSE 8000

WORKDIR /app/src

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]