Example: Aerospace Mechanisms Symposia
======================================
.. contents::
   :local:
   :depth: 2

----

Example: Aerospace Mechanisms Symposia
--------------------------------------
`Aerospace Chatbot, Aerospace Mecahnisms Symposia <https://huggingface.co/spaces/ai-aerospace/aerospace_chatbots>`__

This example uses the deployed Hugging Face model, with Aerospace Mechanisms Symposia papers as input. The `Aerospace Mechanisms Symposia <https://aeromechanisms.com/>`__. There are symposia papers in PDF form going back to the year 1966 with a release every 1-2 years. Each symposia release has roughly 20-40 papers detailing design, test, analysis, and lessons learned for space mechanism design. The full paper index for past symposia is available `here <https://aeromechanisms.com/paper-index/>`__.

The symposia papers are a valuable resource for the aerospace community, but the information is locked in PDF form and not easily searchable. This example demonstrates how to query, search, and ask a Large Language Model (LLM) questions about the content in these papers.

A key feature of this tool is returning source documentation when the LLM provides an answer. This improves trust, enables verification, and allows users to read the original source material.