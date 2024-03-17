Dependencies
============
Dependencies are managed with `poetry <https://python-poetry.org/>`_. Detailed install instructions are located `here <https://www.evernote.com/shard/s84/sh/f37de730-ce37-cd28-789c-86c3dc024a7c/90VLNref38KARua10p4am7IZkwsOxo93fXuBNqba-HpeIkMqGpRZrRkmjw>`_.

* Once poetry is installed, run the following to install all dependencies::

    poetry install

* ``poetry.lock`` and ``pyproject.toml`` are committed to this directory and are the working dependencies.

Install
-------
Follow instructions here: https://python-poetry.org/docs/#installing-with-the-official-installer 

Add to path
-----------
To add the `export PATH="/Users/danmueller/.local/bin:$PATH"` command to your shell configuration file on a macOS system, you can follow these steps:

1. Open Terminal on your Mac. You can find it in the Applications/Utilities folder or use Spotlight to search for "Terminal".

2. In the Terminal window, type the following command to open your shell configuration file in a text editor:
   ````shell
   nano ~/.bash_profile
   ```

   If you're using the Zsh shell, use the following command instead:
   ````shell
   nano ~/.zshrc
   ```

3. The `nano` text editor will open with your shell configuration file. Scroll to the bottom of the file.

4. Add the following line at the end of the file:
   ````shell
   export PATH="/Users/danmueller/.local/bin:$PATH"
   ```

   Note: Replace `/Users/danmueller` with the actual path to your home directory.

5. Press `Ctrl + X` to exit nano. It will prompt you to save the modified buffer. Press `Y` to confirm the changes and then press `Enter` to save the file with the same name.

6. In the Terminal, type the following command to reload your shell configuration:
   ````shell
   source ~/.bash_profile
   ```

   If you're using the Zsh shell, use the following command instead:
   ````shell
   source ~/.zshrc
   ```

   This will apply the changes you made to the shell configuration file.

After following these steps, the `/Users/danmueller/.local/bin` directory will be added to your `PATH` environment variable, allowing you to run the `poetry` command from anywhere in the Terminal.
```

Set up on existing repository from scratch (aerospace_chatbot)
-------------------------------------------------------------
Do this first so that vs code knows where your virtual environments will be (https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option)
```
poetry config virtualenvs.in-project true
```
Run these commands:
- Navigate to local repository
```
cd /aerospace_chatbot
```
- Initialize poetry in the repository for the project. It will ask you for some setup info. 
	- This will give you an error: The current project could not be installed: No file/folder found for package aerospace-chatbot
	- Ignore this error. Your venv will still be set up properly with all of the files required.
```
poetry init
```
- Install poetry in the directory, creates the venv
```
poetry install
```
- Add all of the packages from the repository to poetry (they get added to the pyproject.toml file.
```
cat requirements.txt | xargs poetry add
```
Now these files are created: aerospace_chatbot folder, poetry.lock, pyproject.toml

Setting up from an existing repository with poetry files already defined
-------------------------------------------------------------------------
Once poetry is installed, run the following to install all dependencies: 
```
poetry install
```
- poetry.lock and pyproject.toml are committed to this directory and are the working dependencies.

Set up vs code to run with the venv created
-------------------------------------------
To find where the venv path created is:
```
poetry env info --path
```
It should be in your local directory since you ran the virtualenvs.inproject true command.

Adding packages
---------------
Once you have poetry installed and working in the directory, add packages using poetry with the following line. Once the package is added, commit the poetry.lock and pyproject.toml file.
```
poetry add <package-name>
```

Writing requirements.txt from poetry

$ poetry export --output requirements.txt  