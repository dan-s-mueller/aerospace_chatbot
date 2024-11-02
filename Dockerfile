# Use an official Python runtime as a parent image
FROM python:3.11.5-bookworm

# Do root things: clone repo and install dependencies. libsndfile1 for spotlight. libhdf5-serial-dev for vector distance.
USER root

RUN useradd -m -u 1000 user && chown -R user:user /home/user && chmod -R 777 /home/user

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
RUN mkdir $HOME/config
RUN mkdir $HOME/db
RUN mkdir $HOME/src

# Give all users read/write permissions to the app code directories
RUN chmod 777 $HOME/app
RUN chmod 777 $HOME/config
RUN chmod 777 $HOME/db
RUN chmod 777 $HOME/src

# Set environment variables
ENV PATH="$HOME/.venv/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the rest of your application code.
COPY --chown=user:user ./app $HOME/app
COPY --chown=user:user ./config $HOME/config
COPY --chown=user:user ./db $HOME/db
COPY --chown=user:user ./src $HOME/src

# Set up run env variables.
ENV LOCAL_DB_PATH=$HOME/db
ENV AEROSPACE_CHATBOT_CONFIG='admin'

# Set final work directory for the application
WORKDIR $HOME/app
RUN pwd
RUN ls -R

# Expose the port Streamlit runs on
EXPOSE 8501

# The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# An ENTRYPOINT allows you to configure a container that will run as an executable.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]