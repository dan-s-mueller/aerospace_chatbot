Aerospace Chatbot
=================

.. contents::
   :local:
   :depth: 2

----

Aerospace discipline-specific chatbots and AI tools.

- `Github Repository <https://github.com/dan-s-mueller/aerospace_chatbot>`_
- `Langsmith Project <https://smith.langchain.com/>`_ > Aerospace AI
- `Langsmith Prompts <https://smith.langchain.com/hub/my-prompts?organizationId=45eb8917-7353-4296-978d-bb461fc45c65>`_
- `Hugging Face Organization AI-Aerospace <https://huggingface.co/ai-aerospace>`_

The app can be run in several configurations:

- **Recommended for easiest deployment:** As a streamlit app in the ``src`` directory, running ``streamlit run Home.py``.
- **Recommended for superusers:** With python code following packages and dependencies in the ``src`` directory.
- **Recommended for custom deployments:** As a docker container using the ``Dockerfile`` in the root directory.

Helpful Links for Beginners
---------------------------

- `Hugging Face <https://huggingface.co/>`_, this is where open source models are hosted, also is a great place to deploy applications and perform fine tuning.
- `Langchain <https://python.langchain.com/docs/get_started/introduction>`_, this is a framework for building AI applications.
- `Langsmith <smith.langchain.com>`_, this is a platform for debugging and data analysis for inputs/outputs/runtime of AI models.
- API key links:
    - `OpenAI <https://platform.openai.com/api-keys>`_
    - `Pinecone <https://www.pinecone.io>`_
    - `Hugging Face Hub <https://huggingface.co/settings/tokens>`_
    - `Voyage <https://dash.voyageai.com/api-keys>`_

Running Locally
---------------
It is recommended to run this streamlit app locally for improved performance. The hosted hugging face version is for proof of concept. You must have poetry installed locally to manage depdenencies. To run locally, clone the repository and run the following commands.


.. code-block:: bash

    poetry config virtualenvs.in-project true
    poetry install
    source .venv/bin/activate
    cd ./src
    streamlit run Home.py

Deployed Configurations
-----------------------

A deployed version of this chatbot can be found in a hugging face container called `Aerospace Chatbot <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbots>`__. This version has a pre-configured database and is ready to use. Refer to the `Streamlit Apps <help/streamlit_apps>`_ section for more information about this deployed chatbot.

.. toctree::
    :maxdepth: 2
    :caption: Functionality

    help/dependencies
    help/streamlit_apps
    help/structure
    help/configs

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/api
   indices_and_tables