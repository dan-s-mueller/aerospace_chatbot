User Guide
==========

How to Use
----------
In order for any configuration to work, you'll need the correct dependencies installed. Refer to the :doc:`dependencies` section for more information.

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
    streamlit run Home.py

Deployed Configurations
^^^^^^^^^^^^^^^^^^^^^^^^

A deployed version of this chatbot can be found in a hugging face container called `Aerospace Chatbot <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbots>`__. This version has a pre-configured database and is ready to use under persistent storage. Refer to the `Streamlit Apps <help/streamlit_apps>`_ section for more information.

The dockerfile used to build that space is located `here <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbots/edit/main/Dockerfile>`_.

For specific details of this deployment, refer to the readme here: `Aerospace Chatbot README <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbot_ams/blob/main/README.md>`__.

Using the Backend
^^^^^^^^^^^^^^^^^

You can use the backend code to run the chatbot in a python environment. The backend code is located in the ``src/aerospace_chatbot`` directory, or you can add the package independently via:

.. code-block:: bash

    poetry add aerospace-chatbot

or

.. code-block:: bash

    pip install aerospace-chatbot

Example of how to use the backend are in the ``notebooks`` directory.

Example: Aerospace Mechanisms Symposia
--------------------------------------

`Aerospace Chatbot, Aerospace Mecahnisms Symposia <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbot_ams>`__

This example uses the deployed Hugging Face model, with Aerospace Mechanisms Symposia papers as input. The `Aerospace Mechanisms Symposia <https://aeromechanisms.com/>`__. There are symposia papers in PDF form going back to the year 1966 with a release every 1-2 years. Each symposia release has roughly 20-40 papers detailing design, test, analysis, and lessons learned for space mechanism design. The full paper index for past symposia is available `here <https://aeromechanisms.com/paper-index/>`__.

The symposia papers are a valuable resource for the aerospace community, but the information is locked in PDF form and not easily searchable. This example demonstrates how to query, search, and ask a Large Language Model (LLM) questions about the content in these papers.

A key feature of this tool is returning source documentation when the LLM provides an answer. This improves trust, enables verification, and allows users to read the original source material.

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