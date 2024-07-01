# Use an official Python runtime as a parent image
FROM python:3.11.5-bookworm

# Do root things: clone repo and install dependencies. libsndfile1 for spotlight. libhdf5-serial-dev for vector distance.
USER root

RUN useradd -m -u 1000 user && chown -R user:user /home/user && chmod -R 777 /home/user

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

# Create directories for the app code to be copied into
RUN mkdir $HOME/app 
RUN mkdir $HOME/src
RUN mkdir $HOME/data
RUN mkdir $HOME/config

# Give all users read/write permissions to the app code directories
RUN chmod 777 $HOME/app
RUN chmod 777 $HOME/src
RUN chmod 777 $HOME/data
RUN chmod 777 $HOME/config

# Install Poetry
RUN pip3 install poetry==1.7.1

# Copy poetry files. Use cp for github config, followed by chown statements.
COPY --chown=user:user pyproject.toml $HOME
# RUN cp /clonedir/pyproject.toml $HOME
# RUN chown user:user $HOME/pyproject.toml

# Disable virtual environments creation by Poetry. Making this false results in permissions issues.
RUN poetry config virtualenvs.in-project true

# Set the name of the virtual environment
RUN poetry config virtualenvs.path $HOME/.venv

# Set environment variables
ENV PATH="$HOME/.venv/bin:$PATH"

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of your application code.  Use cp for github config, followed by chown statements. cp commands for non-local builds.
COPY --chown=user:user ./src $HOME/src
COPY --chown=user:user ./data $HOME/data
COPY --chown=user:user ./config $HOME/config
COPY --chown=user:user ./app $HOME/app
# RUN cp -R /clonedir/src /clonedir/data /clonedir/config /clonedir/app $HOME
# RUN chown -R user:user $HOME/src $HOME/data $HOME/config $HOME/db

# Set up database path and env variabole. Comment out if running on hugging face spaces
RUN mkdir $HOME/db
RUN chmod 777 $HOME/db
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
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Run this if you're running with terminal locally
# ENTRYPOINT ["/bin/bash", "-c"]

# To build and run locally
# docker build -t aerospace-chatbot .
# docker run --user 1000:1000 -p 8501:8501 -p 9000:9000 -it aerospace-chatbot

# To build for remote hosts with different architectures:
# docker build --platform linux/amd64 -t dsmuellerpro/aerospace-chatbot:linux-amd64 .

# To run locally with a terminal.
# docker build -t aerospace-chatbot .
# docker run --user 1000:1000 --entrypoint /bin/bash -it aerospace-chatbot

# To run remotely from hugging face spaces
# docker run -it --user 1000:1000 -p 7860:7860 --platform=linux/amd64 \
# 	registry.hf.space/ai-aerospace-aerospace-chatbots:latest