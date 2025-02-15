FROM python:3.11.5-bookworm

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    libsndfile1 \
    curl \
    git \   
    && rm -rf /var/lib/apt/lists/*

# Create user and set up directories
RUN useradd -m -u 1000 user && chown -R user:user /home/user && chmod -R 777 /home/user

# Switch to user
USER user

# Set home and working directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME

# Create and set permissions for directories
RUN mkdir -p $HOME/app $HOME/config $HOME/db $HOME/src && \
    chmod 777 $HOME/app $HOME/config $HOME/db $HOME/src

# Set environment variables
ENV PATH="$HOME/.venv/bin:$PATH"

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create true \
    && poetry config cache-dir /home/user/.cache/poetry \
    && poetry config virtualenvs.path /home/user/.local/share/virtualenvs

# Clone repo and set up project with poetry
RUN git clone https://github.com/dan-s-mueller/aerospace_chatbot.git /tmp/aerospace_chatbot && \
    cp /tmp/aerospace_chatbot/pyproject.toml /tmp/aerospace_chatbot/poetry.lock ./ && \
    cp -r /tmp/aerospace_chatbot/app/* $HOME/app/ && \
    cp -r /tmp/aerospace_chatbot/config/* $HOME/config/ && \
    cp -r /tmp/aerospace_chatbot/src/* $HOME/src/ && \
    python3 -m pip install --user --no-warn-script-location poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi && \
    rm -rf /tmp/aerospace_chatbot

# Set up run env variables
ENV LOCAL_DB_PATH=$HOME/db
ENV PYTHONPATH=$HOME/src:$PYTHONPATH

# Set final work directory
WORKDIR $HOME/app

# Expose port run streamlit
CMD streamlit run app.py \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.fileWatcherType none