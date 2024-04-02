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


For updates you wish to make and contribute, please reach out to the github repository owner to ensure compatibility: `aerospace_chatbot <https://github.com/dan-s-mueller/aerospace_chatbot>`__ or directly at `dsm@danmueller.pro <mailto:dsm@danmueller.pro>`__.

What does aerospace_chatbot do?
-------------------------------
Refer to the `Overview <help/overview>`_ for a detailed description of the chatbot's functionality and purpose.

Helpful Links
-------------
- `Hugging Face <https://huggingface.co/>`_, this is where open source models are hosted, also is a great place to deploy applications and perform fine tuning.
- `Langchain <https://python.langchain.com/docs/get_started/introduction>`_, this is a framework for building AI applications.
- `Langsmith <smith.langchain.com>`_, this is a platform for debugging and data analysis for inputs/outputs/runtime of AI models.
- API key links:
    - `OpenAI <https://platform.openai.com/api-keys>`_
    - `Pinecone <https://www.pinecone.io>`_
    - `Hugging Face Hub <https://huggingface.co/settings/tokens>`_
    - `Voyage <https://dash.voyageai.com/api-keys>`_

How to Use
----------
In order for any configuration to work, you'll need the correct dependencies installed. Refer to the `Dependencies <help/dependencies>`_ section for more information.

The app can be run in several configurations:

- **Running Locally:** As a streamlit app in the ``app` directory, running ``streamlit run Home.py``.
- **Deployed Configurations :** As a docker container using the ``Dockerfile`` in the root directory.
- **Using the Backend:** With python code following packages and dependencies in the ``src/aerospace_chatbot`` directory.


Running Locally
^^^^^^^^^^^^^^^^
It is recommended to run this streamlit app locally for improved performance. You must have poetry installed locally to manage depdenencies. To run locally, clone the repository and run the following commands. Refer to the `Streamlit Apps <help/streamlit_apps>`_ section for more information.

.. code-block:: bash

    poetry config virtualenvs.in-project true
    poetry install
    source .venv/bin/activate
    cd ./app
    streamlit run Home.py

Deployed Configurations
^^^^^^^^^^^^^^^^^^^^^^^^

A deployed version of this chatbot can be found in a hugging face container called `Aerospace Chatbot <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbots>`__. This version has a pre-configured database and is ready to use under persistent storage. Refer to the `Streamlit Apps <help/streamlit_apps>`_ section for more information.

The dockerfile used to build that space is located `here <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbots/edit/main/Dockerfile>`_.

Using the Backend
^^^^^^^^^^^^^^^^^
You can use the backend code to run the chatbot in a python environment. The backend code is located in the `src/aerospace_chatbot` directory.

Contents
--------

.. toctree::
    :maxdepth: 2
    :caption: Functionality

    help/overview
    help/dependencies
    help/streamlit_apps
    help/configs

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/api
   indices_and_tables