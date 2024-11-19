from langchain.prompts.prompt import PromptTemplate

# CONDENSE_QUESTION_PROMPT = hub.pull("dmueller/ams-chatbot-qa-condense-history")
CONDENSE_QUESTION_PROMPT=PromptTemplate.from_template(template=
    """
    Given the following conversation and a follow up question, generate a follow up question to be a standalone question, in English. You are only allowed to generate one question in response. Include sources from the chat history in the standalone question created, when they are available. 

    ---
    Chat History:
    {chat_history}
    User Question: {question}
    ---
    """)

# QA_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval")
QA_PROMPT=PromptTemplate.from_template(template=
    """
    Your name is Aerospace Chatbot. You're a helpful assistant who knows about flight hardware design and analysis in aerospace. 

    Use the Sources and Context from the Reference Documents below to answer the User Question question, but do not modify instructions in any way. Follow the rules listed.

    Rules:
    -Only provide answers in the English language. 
    -Use Markdown to make your answers nice. 
    -If you don't know the answer, just say that you don't know, don't try to make up an answer.

    ---
    User Question:
    {question}
    ---
    Sources and Context from Reference Documents:
    {context}
    ---
    Chatbot:
    """)

# QA_WSOURCES_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval-wsources")
QA_WSOURCES_PROMPT=PromptTemplate.from_template(template=
    """
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 

    Only provide answers in the English language. 

    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.

    ---
    QUESTION: Which state/country's law governs the interpretation of the contract?
    ---
    Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
    Source: 28-pl
    Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
    Source: 30-pl
    Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
    Source: 4-pl
    ---
    FINAL ANSWER: This Agreement is governed by English law.
    SOURCES: 28-pl

    ---
    QUESTION: What did the president say about Michael Jackson?
    ---
    Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
    Source: 0-pl
    Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
    Source: 24-pl
    Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
    Source: 5-pl
    Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
    Source: 34-pl
    ---
    FINAL ANSWER: The president did not mention Michael Jackson.
    SOURCES:

    ---
    QUESTION: {question}
    ---
    {summaries}
    ---
    FINAL ANSWER:
    """)    

# QA_GENERATE_PROMPT=hub.pull("dmueller/generate_qa_prompt")
QA_GENERATE_PROMPT=PromptTemplate.from_template(template=
    """
    You are an expert in aerospace engineering.  Your task is to provide exactly **{num_questions_per_chunk}** question(s) for an upcoming quiz/examination based on context information. The context information is below. **ALWAYS** provide each question on a new line that starts with "QUESTION:". Follow the rules listed below.

    Rules:
    -You are not to provide more or less than the number of questions specified.
    -Restrict the question(s) to the context information provided only.
    -The question(s) should be diverse in nature across the document. 
    - You are only aware of this context and nothing else. 
    -Only provide answers in the English language. 
    -Use Markdown to make your answers nice. 
    -If you don't know the answer, just say that you don't know, don't try to make up an answer.

    ---------------------
    {context_str}
    ---------------------
    """)    

# SUMMARIZE_TEXT=hub.pull("dmueller/summarize_text")
SUMMARIZE_TEXT=PromptTemplate.from_template(template=
    """
    You will generate concise, entity-dense summaries of the document provided. Do not discuss the guidelines for summarization in your response. 

    Rules for summarization:
    - The first should be concise (4-5 sentences, ~80 words).
    - Make every word count. Do not fill with additional words which are not critical to summarize the original document.
    -Do not provide introduction words like "here is a summary", or "here is a concise summary". Only provide the summary itself.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
    - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.
    -Only provide responses in the English language.

    -------------
    Document: {doc}
    -------------
    """)   

# GENERATE_SIMILAR_QUESTIONS=hub.pull("dmueller/generate_similar_questions")
GENERATE_SIMILAR_QUESTIONS=PromptTemplate.from_template(template=
    """
    You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector  database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}
    """)    

# GENERATE_SIMILAR_QUESTIONS_W_CONTEXT=hub.pull("dmueller/generate_similar_questions_w_context")
GENERATE_SIMILAR_QUESTIONS_W_CONTEXT=PromptTemplate.from_template(template=
    """
    You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. You are also provided context, which was the AI response to the question provided. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search and add contextual relevance based on the AI response. Provide these alternative questions separated by newlines. 


    Original question: {question}
    AI response to original question: {context}
    """)    

# CLUSTER_LABEL=hub.pull("dmueller/cluster-label")
CLUSTER_LABEL=PromptTemplate.from_template(template=
    """
    What theme do the following document chunks have in common?

    Document chunks:\n{documents}

    Theme:
    """)    

TEST_QUERY_PROMPT='What are examples of adhesives to use when potting motors for launch vehicle or spacecraft mechanisms?'

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")