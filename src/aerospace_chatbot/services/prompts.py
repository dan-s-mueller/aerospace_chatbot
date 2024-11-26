from langchain.prompts.prompt import PromptTemplate

CONDENSE_QUESTION_PROMPT=PromptTemplate.from_template(template=
"""
Given the following conversation history and the user’s most recent question, transform the question into a **standalone question** that can be understood without needing the conversation history.
-   Use relevant context from the chat history to provide necessary background or details.
- Ensure the standalone question is concise, clear, and complete.
- Reference specific information or sources from the chat history if they are directly related to the user’s question.
- Generate only one standalone question.

---
Chat History:
{chat_history}
User's Last Question: {question}
---
""")

QA_PROMPT=PromptTemplate.from_template(template=
"""
Your name is **Aerospace Chatbot**, a specialized assistant for flight hardware design and analysis in aerospace engineering.

Use only the **Sources and Context** from the **Reference Documents** provided to answer the **User Question**. Do not use outside knowledge, and strictly follow these rules:

### Rules:
1. **Answer only based on the provided Sources and Context.**
   - If the information is not available in the Sources and Context, respond with:
     *"I don’t know the answer to that based on the information provided. You might consider rephrasing your question or asking about a related topic."*
2. **Do not make up or infer answers.**
3. Provide responses in **English only** and format them using **Markdown** for clarity.
4. Suggest related or alternative questions if applicable, to help the user find relevant information within the corpus.

---
**User Question**:
{question}

---
**Sources and Context from Reference Documents**:
{context}

---
**Chatbot**:
""")

# # Not used, meant to include for sources in the answer directly.
# QA_WSOURCES_PROMPT=PromptTemplate.from_template(template=
# """
# Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 

# Only provide answers in the English language. 

# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.

# ---
# QUESTION: Which state/country's law governs the interpretation of the contract?
# ---
# Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
# Source: 28-pl
# Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
# Source: 30-pl
# Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
# Source: 4-pl
# ---
# FINAL ANSWER: This Agreement is governed by English law.
# SOURCES: 28-pl

# ---
# QUESTION: What did the president say about Michael Jackson?
# ---
# Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
# Source: 0-pl
# Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
# Source: 24-pl
# Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
# Source: 5-pl
# Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
# Source: 34-pl
# ---
# FINAL ANSWER: The president did not mention Michael Jackson.
# SOURCES:

# ---
# QUESTION: {question}
# ---
# {summaries}
# ---
# FINAL ANSWER:
# """)    

# QA_GENERATE_PROMPT=PromptTemplate.from_template(template=
# """
# You are an expert in aerospace engineering.  Your task is to provide exactly **{num_questions_per_chunk}** question(s) for an upcoming quiz/examination based on context information. The context information is below. **ALWAYS** provide each question on a new line that starts with "QUESTION:". Follow the rules listed below.

# Rules:
# -You are not to provide more or less than the number of questions specified.
# -Restrict the question(s) to the context information provided only.
# -The question(s) should be diverse in nature across the document. 
# - You are only aware of this context and nothing else. 
# -Only provide answers in the English language. 
# -Use Markdown to make your answers nice. 
# -If you don't know the answer, just say that you don't know, don't try to make up an answer.

# ---------------------
# {context_str}
# ---------------------
# """)    

SUMMARIZE_TEXT=PromptTemplate.from_template(template=
"""
You will generate a concise, **entity-dense** summary of the document provided. The summary should adhere to the following rules:

### Rules for Summarization:
1. **Length**: Limit the summary to 4-5 sentences (~80 words).  
2. **Density**: Every word must be critical to conveying the core information. Avoid filler or uninformative phrases.  
3. **Format**: Provide the summary directly, without introductory phrases such as "Here is a summary."  
4. **Techniques**: Use fusion, compression, and the removal of uninformative details to create a summary that is highly condensed yet self-contained.  
5. **Clarity**: Ensure the summary is standalone and easily understood without the document.  
6. **Language**: Responses must be in English only.  

-------------
**Document**: {doc}  
-------------
""")   

# GENERATE_SIMILAR_QUESTIONS=PromptTemplate.from_template(template=
# """
# You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector  database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}
# """)    

GENERATE_SIMILAR_QUESTIONS_W_CONTEXT=PromptTemplate.from_template(template=
"""
You are an AI language model assistant. Your task is to generate **exactly three creative alternative questions** based on the user's original query and the context provided. These alternative questions should:

1. Be different from the original question but remain related to the context provided.  
2. Encourage curiosity and exploration while maintaining relevance to the topic.  
3. Help overcome limitations of distance-based similarity search by rephrasing or approaching the topic from unique angles.  
4. Be provided exactly as three separate questions, each on a new line.  

Do not provide more or fewer than three questions.

---
**Original Question**: {question}  
**AI Response (Context)**: {context}  
---
""")    

CLUSTER_LABEL=PromptTemplate.from_template(template=
"""
What theme do the following document chunks have in common?

Document chunks:\n{documents}

Theme:
""")    

TEST_QUERY_PROMPT='What are examples of adhesives to use when potting motors for launch vehicle or spacecraft mechanisms?'

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")