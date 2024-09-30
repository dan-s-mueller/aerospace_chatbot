Aerospace Chatbot
=================

.. toctree::
    :hidden:
    :caption: Get Started
    :titlesonly:

    help/overview
    help/user_guide
    help/evaluation

.. toctree::
    :hidden:
    :caption: API Reference
    :glob:
    :titlesonly:

    modules/api
    indices_and_tables

.. rst-class:: lead

   Aerospace discipline-specific chatbots and AI tools.

Deployed apps
-------------

See :doc:`help/deployments`.

Project links
-------------

- `Github Repository <https://github.com/dan-s-mueller/aerospace_chatbot>`_
- `Langsmith Project <https://smith.langchain.com/>`_ > HF_Aerospace_Chatbot_AMS
- `Langsmith Prompts <https://smith.langchain.com/hub/my-prompts?organizationId=45eb8917-7353-4296-978d-bb461fc45c65>`_


What does aerospace_chatbot do?
-------------------------------

Refer to the :doc:`help/overview` for a detailed description of the chatbot's functionality and purpose.

A capability matrix is also located in :doc:`help/user_guide`


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