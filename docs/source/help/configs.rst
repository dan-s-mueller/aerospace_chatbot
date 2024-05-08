Configs
=======

There are standard configuration files used for setup. These are located in the `configs <https://github.com/dan-s-mueller/aerospace_chatbot/tree/main/config>`__ directory. 

`config.json`
-------------
The `config.json` file is the main configuration file used to list available databases, and large language models. Modifying this config file will require ensuring that the models and sources are compatible with the ``aerospace_chatbot`` code. Any changes to the config file should be tested to ensure that the chatbot still functions as expected. 

This file controls the specific models avaialble for embeddings and llms. Submit an issue if you would like to add a new model to the chatbot.

`users.yml`
-----------
This lists available users who can access database processing. When run locally, you can add your own user to this file using instructions `here <https://github.com/mkhorasani/Streamlit-Authenticator>`__.

If you wish to be added as a user to the main repository, reach out to the github repository owner or at `dsm@danmueller.pro <mailto:dsm@danmueller.pro>`__.
