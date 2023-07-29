FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:$PATH"

COPY ./pyproject.toml ./poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY ./app .env /app/
COPY ./utils /app/utils/