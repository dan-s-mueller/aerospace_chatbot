from langchain.prompts.prompt import PromptTemplate

CONDENSE_QUESTION_PROMPT=PromptTemplate.from_template(template=
"""
Given the following conversation history and the user’s most recent question, generate one or more **standalone questions** that incorporate relevant context from the chat history. Always include the user’s exact question as the first standalone question. For additional questions, provide a **detailed summary of the relevant context** from the chat history to ensure the questions are well-informed and self-contained.

### Guidelines:
1. **Include the Exact User Question**: The user’s original question must be included as the first standalone question, unaltered.
2. **Provide Detailed Context for Additional Questions**: Summarize key details from the chat history that are relevant to the user’s question. Use paragraph-length summaries if necessary to capture important details.
3. **Generate Additional Questions (If Applicable)**: If the user’s last question allows for multiple related questions, generate all relevant ones, each accompanied by its own detailed context.
4. **Ensure Self-Containment**: Each question and its context should stand alone without requiring the full chat history.
5. **Clarity and Precision**: Ensure both the context summaries and questions are clear, concise, and directly tied to the user’s last question.
6. **Clean Output Format**: Present the user’s original question first, followed by additional standalone questions, each with its accompanying context.

---
**Chat History**:  
{chat_history}

---
**User's Last Question**: {question}
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

---
**Sources and Context from Reference Documents**:
{context}
---

---
**Chatbot**:
---
""")

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

---
**Document**: {doc}  
---
""")   

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
---

---
**AI Response (Context)**: {context}  
---
""")    

CLUSTER_LABEL=PromptTemplate.from_template(template=
"""
What theme do the following document chunks have in common?

---
**Document chunks**:
{documents}
---

---
**Theme**:
""")    

TEST_QUERY_PROMPT='What are examples of adhesives to use when potting motors for launch vehicle or spacecraft mechanisms?'

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")