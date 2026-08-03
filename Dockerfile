FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install '.[server]'

RUN useradd --create-home --uid 10001 chunkkit
USER chunkkit
EXPOSE 8000
CMD ["chunkkit", "serve", "--host", "0.0.0.0", "--port", "8000"]
