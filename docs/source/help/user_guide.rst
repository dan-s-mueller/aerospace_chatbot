User Guide
==========

.. toctree::
    :hidden:
    :titlesonly:

    functionality
    deployments
    configs

How to Access
-------------
Aerospace Chatbot can be accessed in several configurations:

- **Running Locally:** As a streamlit app in the ``app`` directory, running ``streamlit run app.py``.
- **Deployed Configurations :** See :doc:`deployments`. These contain pre-set settings deployed from containers for easy distribution.
- **Using the Backend:** With python code following packages and dependencies in the ``src/aerospace_chatbot`` directory.

Running Locally
^^^^^^^^^^^^^^^^

Running locally unlocks the most features avaialble to the user. You must have poetry installed locally to manage depdenencies. To run locally, clone the repository and run the following commands. Refer to the `Streamlit Apps <help/streamlit_apps>`_ section for more information.

.. code-block:: bash

    poetry config virtualenvs.in-project true
    poetry install
    source .venv/bin/activate
    cd ./app
    streamlit run app.py

Deployed Configurations
^^^^^^^^^^^^^^^^^^^^^^^^

For deployed versions, refer to :doc:`deployments`.


Using the Backend
^^^^^^^^^^^^^^^^^

You can use the backend code to run the chatbot in a python environment. The backend code is located in the ``src/aerospace_chatbot`` directory, or you can add the package independently via:

.. code-block:: bash

    poetry add aerospace-chatbot

or

.. code-block:: bash

    pip install aerospace-chatbot

Example of how to use the backend are in the ``notebooks`` directory.

Compatibility Matrix
--------------------

This matrix gives options for the user, depending on what the objective is. Proprietary data sources likely want air gapped solutions run locally, vs. collaborative teams with non-proprietary sources likely want remotely hosted options.

+-------------------------+-----------------------+------------------------------------------------+----------------------+
| Feature                 | Index Type            | Embedding Model                                | LLM                  |
+=========================+=======================+================================================+======================+
| Remotely hosted         | Pinecone              | colbert-ir/colbertv2.0 (Hugging Face RAG Model)| OpenAI, Hugging Face |
+-------------------------+-----------------------+------------------------------------------------+----------------------+
| Open source             | ChromaDB, RAGatouille | colbert-ir/colbertv2.0 (Hugging Face RAG Model)| Hugging Face         |
+-------------------------+-----------------------+------------------------------------------------+----------------------+
| Proprietary             | Pinecone              | OpenAI, Voyage                                 | OpenAI               |
+-------------------------+-----------------------+------------------------------------------------+----------------------+
| Air gapped (fully local)| ChromaDB, RAGatouille | colbert-ir/colbertv2.0 (Hugging Face RAG Model)| LM Studio            |
+-------------------------+-----------------------+------------------------------------------------+----------------------+
| RAG type: Standard      | Any                   | Any                                            | Any                  |
+-------------------------+-----------------------+------------------------------------------------+----------------------+
| RAG type: Parent-Child  | ChromaDB, Pinecone    | OpenAI, Voyage                                 | Any                  |
+-------------------------+-----------------------+------------------------------------------------+----------------------+
| RAG type: Summary       | ChromaDB, Pinecone    | OpenAI, Voyage                                 | Any                  |
+-------------------------+-----------------------+------------------------------------------------+----------------------+