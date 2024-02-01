# Use an official Python runtime as a parent image
FROM python:3.11.5-bookworm

# The next few lines come from here: https://huggingface.co/docs/hub/spaces-sdks-docker#permissions
# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/data \
	PATH=/data/.local/bin:$PATH

# Change permissions for to read/write/execute
# RUN mkdir $HOME/app
# RUN chmod 777 $HOME/app

# Set the working directory to the user's home directory
WORKDIR $HOME

# Install poetry
RUN pip3 install poetry

# Copy only the necessary files for installing dependencies
COPY --chown=user pyproject.toml poetry.lock ./

# Disable virtual environments creation by Poetry
# as the Docker container itself is an isolated environment
RUN poetry config virtualenvs.in-project true

# Set the name of the virtual environment
RUN poetry config virtualenvs.path $HOME/.venv

#  Install dependencies
RUN poetry install

# Set the .venv folder as the virtual environment directory
ENV PATH="$HOME/.venv/bin:$PATH"

# Copy the current directory contents into the container
COPY --chown=user . $HOME

# Make a port available to the world outside this container
# The EXPOSE instruction informs Docker that the container listens on the specified network ports at runtime. Your container needs to listen to Streamlit’s (default) port 8501.
EXPOSE 8501

# The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501:
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Update working directory to be consistent with where Start.py is
WORKDIR $HOME/scripts

# An ENTRYPOINT allows you to configure a container that will run as an executable. Here, it also contains the entire streamlit run command for your app, so you don’t have to call it from the command line
ENTRYPOINT ["streamlit", "run", "Start.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Execute with:
# docker build -t <image_name> .    
# docker run -p 8501:8501 <image_name>