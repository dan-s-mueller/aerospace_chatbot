# Use an official Python runtime as a parent image
FROM python:3.11.5-bookworm

# Came from here: https://huggingface.co/docs/hub/spaces-sdks-docker#permissions Set up a new user named "user" with user ID 1000.
RUN useradd -m -u 1000 user

# Do root things: clone repo and install dependencies
USER root
# WORKDIR /clonedir
# RUN apt-get update && \
# 	apt-get install -y git
# RUN git clone --depth 1 https://github.com/dan-s-mueller/aerospace_chatbot.git .
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME

# Create directories for the app code to be copied into
# RUN mkdir $HOME/app
# RUN mkdir $HOME/src 
# RUN mkdir $HOME/data 
# RUN mkdir $HOME/config

# Install Poetry
RUN pip3 install poetry==1.7.1

# Copy poetry files
COPY --chown=user pyproject.toml $HOME

# Disable virtual environments creation by Poetry. Making this false results in permissions issues.
RUN poetry config virtualenvs.in-project true

# Set the name of the virtual environment
RUN poetry config virtualenvs.path $HOME/.venv

# Set environment variables
ENV PATH="$HOME/.venv/bin:$PATH"

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of your application code
# COPY --chown=user . $HOME
COPY ./src $HOME/src
COPY ./data $HOME/data
COPY ./config $HOME/config
COPY ./app $HOME/app

# Expose the port Streamlit runs on
EXPOSE 8501

# The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501:
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set final work directory for the application
WORKDIR $HOME/app

RUN ls -R

# An ENTRYPOINT allows you to configure a container that will run as an executable.
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