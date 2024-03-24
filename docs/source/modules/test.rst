Test
============

.. automodule:: tests.test
   :members:
   :undoc-members:
   :show-inheritance:


Testing is performed for backend (modules) and frontend (streamlit) separately. All test are located in the ``tests`` directory. Before pulling branches into main, please ensure that all tests run successfully.

Backend Testing
----------------
Run the following for backend testing:

.. code-block:: shell

   python -m unittest test_backend.py


Frontend Testing
-----------------
Run the following for frontend testing:

.. code-block:: shell

   streamlit run test_frontend.py