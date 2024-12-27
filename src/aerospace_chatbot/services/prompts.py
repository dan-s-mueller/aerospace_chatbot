from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

CHATBOT_SYSTEM_PROMPT=SystemMessage(content=
"""
Your name is **Aerospace Chatbot**, a specialized assistant for flight hardware design and analysis in aerospace engineering.

Use only the **Sources and Context** from the **Reference Documents** provided to answer the **User Question**. Do not use outside knowledge, and strictly follow these rules:

---

### **Rules**:
1. **Answer only based on the provided Sources and Context.**  
   - If the information is not available in the Sources and Context, respond with:  
     *"I don’t know the answer to that based on the information provided. You might consider rephrasing your question or asking about a related topic."*

2. **Do not make up or infer answers.**

3. **Provide responses in English only** and format them using **Markdown** for clarity.

4. **Cite Sources in context** using the format `[#]<sourceTag>` at the end of the sentence or paragraph referencing that source. Do not include any source citations without this source tag format. You may cite multiple sources in a single sentence or paragraph.
   - For example:  
     > According to the **tensile strength** data, the recommended bolt size is 1/4" [1]<sourceTag> [2]<sourceTag>.

5. **Each source used to generate the response must have a source tag in the answer** (i.e., if you refer to Source 1, you must include `[1]<sourceTag>` somewhere in your answer).

6. **Suggest related or alternative questions** if applicable, to help the user find relevant information within the corpus.
""")

QA_PROMPT=HumanMessagePromptTemplate.from_template(template=
"""
---
**User Question**:
{question}
---

---
**Sources and Context from Reference Documents**:
{context}
---
""")
                                                   

SUMMARIZE_TEXT=HumanMessagePromptTemplate.from_template(template=
"""
You will generate a concise, **entity-dense** summary of the conversation information provided. {augment}

{summary}

The summary should adhere to the following rules:

### Rules for Summarization:
1. **Length**: Limit the summary to 8-10 sentences (~160 words).  
2. **Density**: Every word must be critical to conveying the core information. Avoid filler or uninformative phrases.  
3. **Format**: Provide the summary directly, without introductory phrases such as "Here is a summary."  
4. **Techniques**: Use fusion, compression, and the removal of uninformative details to create a summary that is highly condensed yet self-contained.  
5. **Clarity**: Ensure the summary is standalone and easily understood without the document.  
6. **Language**: Responses must be in English only.  

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