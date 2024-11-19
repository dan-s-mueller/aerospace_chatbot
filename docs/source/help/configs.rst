Configs
=======

There are standard configuration files used for setup. These are located in the `configs <https://github.com/dan-s-mueller/aerospace_chatbot/tree/main/config>`__ directory. 

`config.json`
-------------
The `config.json` file is the main configuration file used to list available databases, and large language models. Modifying this config file will require ensuring that the models and sources are compatible with the ``aerospace_chatbot`` code. Any changes to the config file should be tested to ensure that the chatbot still functions as expected. 

This file controls the specific models avaialble for embeddings and llms. Submit an issue on github if you would like to add a new model to the chatbot.

Environment Variables
---------------------

There are several environment variables that are used to control the chatbot's behavior. These can be set in the `.env` file, or if in a deployed environment statically.

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `PINECONE_API_KEY`, `HUGGINGFACEHUB_API_TOKEN` - These are the API keys used to access the services.
- `LOCAL_DB_PATH` - This is the path to the local database used for storing chat.
- `AEROSPACE_CHATBOT_CONFIG` - This sets the level of access for the chatbot. `admin` provides full functionality. `tester` limits functionality to fewer models and read only access to the databases.
- `LOG_LEVEL` - This is the level of logging to use.
- `LOG_FILE` - This is the path to the log file.
- `LOG_FORMAT` - This is the format of the log file.