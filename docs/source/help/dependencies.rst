Dependencies
============
.. contents::
   :local:
   :depth: 2

----

The main files in the repository required for dependencies are:

- ``poetry.lock`` and ``pyproject.toml`` to define all dependencies

- ``poetry install`` to install all dependencies

The instructions are tailored to unix-based systems, modify as needed for Windows.

Install
-------
Follow instructions `here <https://python-poetry.org/docs/#installing-with-the-official-installer>`__.

Add to path
```````````
To add the ``export PATH="/Users/danmueller/.local/bin:$PATH"`` command to your shell configuration file on a macOS system, you can follow these steps:

1. Open Terminal. You can find it in the Applications/Utilities folder or use Spotlight to search for "Terminal".

2. In the Terminal window, type the following command to open your shell configuration file in a text editor:

   .. code-block:: shell

      nano ~/.bash_profile

   If you're using the Zsh shell, use the following command instead:

   .. code-block:: shell

      nano ~/.zshrc

3. The ``nano`` text editor will open with your shell configuration file. Scroll to the bottom of the file.

4. Add the following line at the end of the file:

   .. code-block:: shell

      export PATH="/Users/danmueller/.local/bin:$PATH"

   Note: Replace ``/Users/danmueller`` with the actual path to your home directory.

5. Press ``Ctrl + X`` to exit nano. It will prompt you to save the modified buffer. Press ``Y`` to confirm the changes and then press ``Enter`` to save the file with the same name.

6. In the Terminal, type the following command to reload your shell configuration:

   .. code-block:: shell

      source ~/.bash_profile

   If you're using the Zsh shell, use the following command instead:

   .. code-block:: shell

      source ~/.zshrc

   This will apply the changes you made to the shell configuration file.

After following these steps, the ``/Users/danmueller/.local/bin`` directory will be added to your ``PATH`` environment variable, allowing you to run the ``poetry`` command from anywhere in the Terminal.

Setup Types
-----------

Existing Repository Setup with Poetry Files
````````````````````````````````````````````````````````````````````````
Once poetry is installed, run the following to install all dependencies. ``no-root`` is used to not install the current project as a package.

.. code-block:: shell

   poetry install --no-root

``poetry.lock`` and ``pyproject.toml`` are committed to this directory and are the working dependencies.

Other Things
------------
Adding packages
```````````````````````````
Once you have poetry installed and working in the directory, add packages using poetry with the following line. Once the package is added, commit the poetry.lock and pyproject.toml file.

.. code-block:: shell

   poetry add <package-name>

Writing Requirements.txt
````````````````````````

.. code-block:: shell

   poetry export --output requirements.txt

Set up VS Code after Installing Poetry
````````````````````````````````````````````
To find where the venv path created is run:

.. code-block:: shell

   poetry env info --path

It should be in your local directory since you ran the ``virtualenvs.inproject true`` command.
