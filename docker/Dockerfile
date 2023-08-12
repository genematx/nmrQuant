FROM python:3.11.4-slim AS base

ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        python3-dev&& \
    apt clean

WORKDIR /opt

# Upgrade pip and install required packages (numpy needed to build some wheels)
RUN pip3 install --upgrade pip
COPY ./docker/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

FROM base AS prod

# Install the q2nmr package
COPY ./src ./src
RUN pip install ./src

FROM base AS dev

# set up for local development
COPY ./docker/requirements.dev.txt requirements.dev.txt
RUN pip install -r requirements.dev.txt

# Install the q2nmr package in editable mode
COPY ./src ./src
RUN pip install -e ./src

COPY ./docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash", "/entrypoint.sh"]
