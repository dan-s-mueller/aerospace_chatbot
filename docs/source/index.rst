Aerospace Chatbot
=================

.. toctree::
    :hidden:
    :caption: Get Started
    :titlesonly:

    help/functionality
    help/configs

.. toctree::
    :hidden:
    :caption: API Reference
    :maxdepth: 2
    :includehidden:

    modules/core
    modules/processing
    modules/services
    modules/ui
    modules/tests
    indices_and_tables

.. rst-class:: lead

   Aerospace discipline-specific chatbots and AI tools.

Project links
-------------

- `Github Repository <https://github.com/dan-s-mueller/aerospace_chatbot>`_

Contribute
----------

- Bugs or improvements: `create an issue <https://github.com/dan-s-mueller/aerospace_chatbot/issues/new/choose>`__
- Pull requests: `create a pull request <https://github.com/dan-s-mueller/aerospace_chatbot/compare>`__
- Reach out to the github repository owner at `dsm@danmueller.pro <mailto:dsm@danmueller.pro>`__

References
-------------

Models
^^^^^^^

- `Hugging Face <https://huggingface.co/>`_, this is where open source models are hosted, also is a great place to deploy applications and perform fine tuning.
    - `LLM leaderboard <https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard>`__
    - `Embedding leaderboard <https://huggingface.co/spaces/mteb/leaderboard>`__
- `Voyage embedding models <https://docs.voyageai.com/docs/embeddings>`__, this is a proprietary high performance embedding model.
- OpenAI, this is where OpenAI LLMs and embeddings are hostes
    - `OpenAI LLM models <https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4>`__
    - `OpenAI Embedding models <https://platform.openai.com/docs/models/embeddings>`__

Frameworks
^^^^^^^^^^

- `Langchain <https://python.langchain.com/docs/get_started/introduction>`_, this is a framework for building AI applications.
- `Langsmith <https://smith.langchain.com/>`__, this is a platform for debugging and data analysis for inputs/outputs/runtime of AI models.
- `Spotlight from Renumics <https://renumics.com/open-source/spotlight/>`__, open source platform GUI for data visualization and analysis.

API key links
^^^^^^^^^^^^^

- `OpenAI <https://platform.openai.com/api-keys>`__
- `Pinecone <https://www.pinecone.io>`__
- `Hugging Face Hub <https://huggingface.co/settings/tokens>`__
- `Voyage <https://dash.voyageai.com/api-keys>`__

Dependencies
^^^^^^^^^^^^

The main files in the repository required for dependencies are:

- ``poetry.lock`` and ``pyproject.toml`` to define all dependencies

- ``poetry install`` to install all dependencies

The instructions are tailored to unix-based systems, modify as needed for Windows.

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