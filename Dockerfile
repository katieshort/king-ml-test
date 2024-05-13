## Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies required for compiling Python packages
RUN apt-get update && apt-get install -y gcc g++ libpq-dev
#&& apt-get install -y --no-install-recommends curl git gcc libpq-dev \
#

RUN pip install poetry==1.5.1


ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

## Set the working directory in the container
WORKDIR /app

COPY poetry.lock pyproject.toml /app/
RUN touch README.md

# Install dependencies without installing the project itself
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Copy the rest of your application code
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

CMD ["poetry", "run", "python", "app.py"]

