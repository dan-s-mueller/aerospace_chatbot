# Use an official Python runtime as a parent image
FROM python:3.11.5-bookworm

# The next few lines come from here: https://huggingface.co/docs/hub/spaces-sdks-docker#permissions
# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
# RUN pip install --no-cache-dir --upgrade pip

# Install Poetry
RUN pip3 install poetry==1.7.1

# Copy poetry files
COPY --chown=user pyproject.toml poetry.lock* $HOME

# Disable virtual environments creation by Poetry
# as the Docker container itself is an isolated environment
RUN poetry config virtualenvs.in-project true

# Set the name of the virtual environment
RUN poetry config virtualenvs.path $HOME/.venv

# Set environment variables
ENV PATH="$HOME/.venv/bin:$PATH"

# Install dependencies using Poetry
RUN poetry install --no-dev

# Copy the rest of your application code
COPY --chown=user . $HOME

# Expose the port Streamlit runs on
EXPOSE 8501

# The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501:
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Update working directory to be consistent with where Start.py is
WORKDIR $HOME/scripts

# An ENTRYPOINT allows you to configure a container that will run as an executable. Here, it also contains the entire streamlit run command for your app, so you don’t have to call it from the command line
ENTRYPOINT ["streamlit", "run", "Start.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker run -it -p 7860:7860 --platform=linux/amd64 \
# 	registry.hf.space/ai-aerospace-aerospace-chatbots:latest 