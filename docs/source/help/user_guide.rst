User Guide
==========

How to Use
----------
In order for any configuration to work, you'll need the correct dependencies installed. This is controlled through poetry and described on the documentation homepage.

The app can be run in several configurations:

- **Running Locally:** As a streamlit app in the ``app`` directory, running ``streamlit run Home.py``.
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

Advanced functionality
----------------------

Index Types
^^^^^^^^^^^

RAGatouille
"""""""""""

RAGatouille docs are located here:
- `Github Repository <https://github.com/bclavie/RAGatouille>`__
- `API docs <https://ben.clavie.eu/ragatouille/api/#ragatouille.RAGPretrainedModel.RAGPretrainedModel.index>`__

This functionality will create an indexed database using ColBERT late-interaction retrieval. For each document chunk which is uploaded it will take approximately 1-2 seconds. To not exceed context limitations of ColBERT, each document provided will be split into 256 token chunks. 

RAG Types
^^^^^^^^^^

Parent-Child RAG
""""""""""""""""

`Under construction`

Summary RAG
"""""""""""

`Under construction`

Data Visualization
^^^^^^^^^^^^^^^^^^

`Under construction`, see `Spotlight from Renumics <https://renumics.com/open-source/spotlight/>`__