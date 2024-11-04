Functionality
=============
.. note::
  Some functionality may be limited for non-locally deployed applications (e.g. demos), which by default uses `AEROSPACE_CHATBOT_CONFIG=tester`.

Basic Functionality
-------------------
The Streamlit app is initialized from ``app.py`` or via an access link to a pre-deployed container. The sidebar lists features which are used to manage databases, chat with PDF data, and visualize the PDF data. Each of the features, are described below. The available models and functionality is set by the `AEROSPACE_CHATBOT_CONFIG` environment variable.

Sidebar
^^^^^^^
In each app there is a sidebar which is used to apply settings. The sidebar is dynamic and will only show the settings that are relevant to the app. Each selectable item has a hover over help button describing what the item does. For demos, the sidebar has limited functionality.

.. image:: ../images/sidebar.png
  :alt: Home
  :align: center

Options which require further explination are described below. LLM options and types have hyperlinks in the help button in each app.

- RAG Type: Choose between Standard, `Parent-Child <https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever/>`_, and `Summary <https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector/#summary>`__. Refer to these links to read about how these retrievers work.
- Index Selected: When querying the database, this will select the index you interact with. It is useful when you have multiple indexes of different types.

Aerospace Chatbot
^^^^^^^^^^^^^^^^^^
This is the main app which is used for querying and asking questions about the data uploaded from PDFs. Functionality here includes: memory of the previous prompt/queries, alternate questions based on the prompt and response, and source documents. These are available under the dropdown below the last response.

Before the first entry, you have the ability to upload files to the database.

To restart a conversation, click the "Restart session" button twice.

When a query is made, the responses are stored in a separate vector database. This database is used for visualization in the Visualize Data app, which can be used to explore.

PDF segments which are relevant are shown at the bottom and expandable.

.. image:: ../images/chatbot.png
  :alt: Chatbot
  :align: center

Sidebar Options
^^^^^^^^^^^^^^^
.. note::
  This app has restricted fucntionality when used with `AEROSPACE_CHATBOT_CONFIG=tester`. See :doc:`configs` for details.
  Secret keys are not required in demos. Inputting new secret keys will overwrite the existing ones, but are not saved.

.. image:: ../sidebar.png
  :alt: Database Processing
  :align: center

- Index type: Vector database type (e.g. Chroma, Pinecone, etc.)
- Embedding model family: Embedding family type (e.g. OpenAI, Voyage, etc.)
- Embedding model: The specific model chosen from the family (e.g. text-embedding-3-large, voyage-large-2-instruct, etc.)
- RAG type: The type of RAG process used (e.g. Standard, Parent-Child, etc.)

- Index selected: The index selected to be used in the RAG process.
- LLM model: First selects the family of models and then selects from models to be used (e.g. gpt-4o, gpt-4o-mini)
- LLM options: temperature (randomness) and max tokens (length of output)

- Number of items per prompt: The number of text chunks to be used for the context in the response generation during the RAG process. More items will give more context, but requires longer context windows.
- Search type: the method used to determine the most similar vectors in the vector database to your query.
- Secret keys: API keys for models and databases. For demos, this is not required.

Database Processing
^^^^^^^^^^^^^^^^^^^
.. note::
  This app is restricted use with `AEROSPACE_CHATBOT_CONFIG=admin`. See :doc:`configs` for details.

This app is used to manage databases. You can add, delete, and update databases. There are options available to export intermediate files for debugging or further processing.

Deleting existing databases is available via the connection status dropdown.

.. image:: ../images/database_processing.png
  :alt: Database Processing
  :align: center

Visualize Data
^^^^^^^^^^^^^^
.. note::
  This app is restricted use with `AEROSPACE_CHATBOT_CONFIG=admin`. See :doc:`configs` for details.

This app allows visualization using `Spotlight from Renumics <https://renumics.com/open-source/spotlight/>`__. A new browser will open with the visualization. The data is loaded from the database and the settings are applied from the sidebar.

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

`Under construction`