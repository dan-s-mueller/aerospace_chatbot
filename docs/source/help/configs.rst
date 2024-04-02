Configs
=======
.. contents::
   :local:
   :depth: 2

----

There are standard configuration files used for setup. These are located in the `configs <https://github.com/dan-s-mueller/aerospace_chatbot/tree/main/config>`__ directory. 

`config.json`
-------------
The `config.json` file is the main configuration file used to list available databases, and large language models. Modifying this config file will require ensuring that the models and sources are compatible with the `aerospace_chatbot` code. Any changes to the config file should be tested to ensure that the chatbot still functions as expected. 

`index_data.json`
-----------------

A file listing available database types and their corresponding database files. This file is used to quickly load the available databases and their corresponding files. It is not recomended to modify this file unless you are adding a new database type.

`users.yml`

This lists available users who can access database processing. When run locally, you can add your own user to this file using instructions `here <https://github.com/mkhorasani/Streamlit-Authenticator>`__.

If you wish to be added as a user to the main repository, reach out to the github repository owner.