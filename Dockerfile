# Use an official Python runtime as a parent image
FROM python:3.11.5-bookworm

# Came from here: https://huggingface.co/docs/hub/spaces-sdks-docker#permissions Set up a new user named "user" with user ID 1000.
RUN useradd -m -u 1000 user

# Do root things: clone repo and install dependencies. libsndfile1 for spotlight. libhdf5-serial-dev for vector distance.
USER root
# WORKDIR /clonedir
# RUN apt-get update && \
# 	apt-get install -y git
# RUN git clone --depth 1 https://github.com/dan-s-mueller/aerospace_chatbot.git .
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    libsndfile1 \   
    && rm -rf /var/lib/apt/lists/*
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME

# Set the home directory so any user can read/write to it.

# Create directories for the app code to be copied into
# RUN mkdir $HOME/app
# RUN mkdir $HOME/src
# RUN mkdir $HOME/data
# RUN mkdir $HOME/config

RUN mkdir $HOME/app && chmod a+rw $HOME/app
RUN mkdir $HOME/src && chmod a+rw $HOME/src
RUN mkdir $HOME/data && chmod a+rw $HOME/data
RUN mkdir $HOME/config && chmod a+rw $HOME/config

# Check if /data directory exists, if not create a local db directory. Hugging face spaces has it by default, but not local docker.
# RUN if [ ! -d "/data" ]; then \
# 		mkdir $HOME/db; \
# 	fi

# Install Poetry
RUN pip3 install poetry==1.7.1

# Copy poetry files. Use cp for github config, followed by chown statements.
COPY --chown=user pyproject.toml $HOME

# Disable virtual environments creation by Poetry. Making this false results in permissions issues.
RUN poetry config virtualenvs.in-project true

# Set the name of the virtual environment
RUN poetry config virtualenvs.path $HOME/.venv

# Set environment variables
ENV PATH="$HOME/.venv/bin:$PATH"

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of your application code.  Use cp for github config, followed by chown statements.
COPY ./src $HOME/src
COPY ./data $HOME/data
COPY ./config $HOME/config
COPY ./app $HOME/app

# For local deployments.
RUN mkdir $HOME/db && chmod a+rw $HOME/db
# RUN mkdir $HOME/db
# RUN chmod -R 666 $HOME/db
ENV LOCAL_DB_PATH=$HOME/db

# Set final work directory for the application
WORKDIR $HOME/app
RUN pwd
RUN ls -R

# Expose the port Streamlit runs on
EXPOSE 8501

# The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501:
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# An ENTRYPOINT allows you to configure a container that will run as an executable.
# TODO test out that there are not weird bugs when running on docker locally. Spotlight, databases, etc.
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Run this if you're running with terminal locally
# ENTRYPOINT ["/bin/bash", "-c"]

# To run locally
# docker build -t aerospace-chatbot .
# docker run -p 8501:8501 aerospace-chatbot

# To run locally with a terminal.
# docker build -t aerospace-chatbot .
# docker run -it --entrypoint /bin/bash aerospace-chatbot

# To run remotely from hugging face spaces
# docker run -it -p 7860:7860 --platform=linux/amd64 \
# 	registry.hf.space/ai-aerospace-aerospace-chatbots:latest