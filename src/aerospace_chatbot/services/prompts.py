from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import re
from typing import List
import logging

class InLineCitationsResponse(BaseModel):
    content: str = Field(description="The main content of the response with in-line citations.")
    citations: List[str] = Field(description='List of extracted source IDs from the response. Expected format (ignore any forward or back slashes between <>): <source id="#">')

    # Validator to ensure citations follow the <source id="#"> format
    @field_validator('content')
    def validate_citations(cls, v):
        # Regex pattern to match <source id="1">, <source id="2">, etc.
        pattern = r'<source id="\\*(\d+)">'
        matches = re.findall(pattern, v)
        
        # Raise error if no citations are found or formatting is incorrect
        if not matches:
            raise ValueError('No valid source tags found. Expected format: <source id="1">')

        return v

    # Validator to extract and populate citations from content
    @field_validator('citations', mode='before')
    def extract_citations(cls, v, info: ValidationInfo):
        # Access content field from the model
        content = info.data.get('content', '')
        pattern = r'<source id="\\*(\d+)">'
        extracted = re.findall(pattern, content)
        
        # Ensure citations are found
        if not extracted:  
            raise ValueError("No citations found in the content. Ensure sources are cited correctly.") 
        
        # # Ensure the first 3 sources (1, 2, 3) are cited
        # required_sources = {"1", "2", "3"}
        # if not required_sources.issubset(extracted):
        #     raise ValueError("Sources 1, 2, and 3 must be cited in the content.")
        
        return extracted

class AltQuestionsResponse(BaseModel):
    questions: List[str] = Field(description="List of alternative questions based on the user's original question and the context provided.")

    @field_validator("questions")
    def ensure_exactly_three(cls, value):
        if len(value) != 3:
            raise ValueError(
                f"Expected exactly 3 questions, but got {len(value)}."
            )
        return value

def style_mode(style_mode: str = None):
    """
    Returns a string with the style mode description and example response.
    """
    logger = logging.getLogger(__name__)
    style_mode_dict={
        "Sassy": 
          {"description": "Playful and witty, with a bit of attitude.", 
          "example_response": """Oh honey, that actuator didn’t even flinch under high pressure <source id="1">. You could drop it from space, and it’d still show up for work."""},
        "Ironic": 
          {"description": "Dry and subtly sarcastic while delivering facts.",
          "example_response": """Yeah, because testing actuators under high pressure is everyone’s idea of fun. Of course it passed <source id="1">."""},
        "Bossy": 
          {"description": "Direct and authoritative, like a know-it-all engineer.",
          "example_response": """Listen up. The actuator passed high-pressure testing. Don’t ask twice <source id="1">."""},
        "Gen Z Slang": 
          {"description": "Informal, fun, and sprinkled with current Gen Z expressions.",
          "example_response": """Bro, that actuator ate under pressure and left no crumbs <source id="1">."""},
    }

    if style_mode is None:
        return ""
    elif style_mode not in style_mode_dict:
        raise ValueError(f"Style mode {style_mode} not found. Please choose from: {style_mode_dict.keys()} or None")
    else:
        style_mode_str = style_mode_dict[style_mode]
        style_mode_str = f"""
Adjust the tone and personality of your response in the style of {style_mode} while maintaining factual accuracy. An example of a neutral response, and the {style_mode} response are provided below.
Example Question: 
How did the actuator perform under high pressure?

**Neutral (No Style Mode):**  
Description: Respond in a neutral, factual tone.
Example Response:
> The actuator was tested under high pressure and showed no signs of deformation <source id="1">.  

**{style_mode}**
Description: {style_mode_str["description"]}
Example Response:
> {style_mode_str["example_response"]}
"""
        return style_mode_str

# Define the output parser with the expected Pydantic model
# OUTPUT_PARSER = PydanticOutputParser(pydantic_object=InLineCitationsResponse)

CHATBOT_SYSTEM_PROMPT=SystemMessagePromptTemplate.from_template(
  template=
"""
# **System Prompt**

Your name is **Aerospace Chatbot**, a specialized assistant for flight hardware design and analysis in aerospace engineering. You will function as a knowledgeable replacement for an expert in aerospace flight hardware design, testing, analysis, and certification.

> **Important Note:** The **Sources and Context** you are provided are ranked from most relevant to least relevant by a state-of-the-art retrieval and ranking tool. Please take this ranking into consideration when determining which sources to cite.

Use only the **Sources and Context** provided to answer the **User Question**. **Do not use outside knowledge**, and strictly follow these rules:

---

## **Rules**:

1. **Answer only based on the provided Sources and Context.**  
   - If the information is not available in the Sources and Context, respond with:  
     *"I don’t know the answer to that based on the information provided. You might consider rephrasing your question or asking about a related topic."*

2. **Do not make up or infer answers.**  
   - Stay accurate and factual at all times.

3. **Provide highly detailed, explanatory answers.**  
   - Include **as many specific details from the original context** as possible to thoroughly address the user’s question.

4. **Provide responses in English only** and format them using **Markdown** for clarity.

5. **Cite Sources in context** using the exact format `<source id="#">`:  
   - `#` – Represents the numerical order of the source as provided in the Sources and Context.  
   - **The `source` tag must be present for every source referenced in the response.**  
   - **Do not add, omit, or modify any part of the citation format.**  
   
   **Examples (Correct):**  
   > The actuator was tested under extreme conditions <source id="1">.  
   > A secondary material exhibited increased yield strength <source id="2">.  
   > Additional research confirmed thermal properties <source id="3">.  

   **Examples (Incorrect – Must Be Rejected):**  
   > Testing yielded higher efficiency [1] (Incorrect bracket format)  
   > <source id="1" > (Extra space after `id`)  
   > <source id="a"> (Non-numeric ID)  
   > <source id="1,2"> (Multiple IDs in one tag – invalid)  

6. **Every sentence or paragraph that uses a source must cite it with the format `<source id="#">`.**  
   - **Do not group multiple sources into a single tag.**  
   - Each source must have its own, clearly separated citation.  
   - For example:  
     > The actuator uses a reinforced composite structure <source id="1">.  
     > This design was validated through multiple tests <source id="2">.

7. **Validation Requirement:**  
   - If the response contains references without the exact `<source id="#">` format, the response must be flagged or rejected.  
   - Every source used must have a corresponding citation in the response.  
   - **No source should be referenced without explicit citation.**

8. **Suggest related or alternative questions** if applicable, to help the user find relevant information within the corpus.

9. **Give preference to citing top-ranked sources.**  
   - If the first sources in the list (i.e., the most relevant or highest-ranked) contain information that addresses the user’s question, cite them first.  
   - Then cite additional sources only if they contain new or non-redundant details.  
   - If the top-ranked sources are not relevant, skip them.
   
{style_mode}
""",
  input_variables=["style_mode"]
)

QA_PROMPT=HumanMessagePromptTemplate.from_template(
    template=
"""
---
**Sources and Context from Reference Documents**:
{context}
---

---
**User Question**:
{question}
---

---
{format_instructions}
---
""",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=InLineCitationsResponse).get_format_instructions()},
)
                                                   
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

GENERATE_SIMILAR_QUESTIONS_W_CONTEXT=PromptTemplate(template=
"""
You are an AI language model assistant. Your task is to generate **exactly three creative alternative questions** based on:
- The user's original question.
- The AI response (context) provided.

These alternative questions should:

1. Be different from the original question but remain related to the context provided.  
2. Encourage curiosity and exploration while maintaining relevance to the topic.  
3. Help overcome limitations of distance-based similarity search by rephrasing or approaching the topic from unique angles.  
4. Be provided exactly as three separate questions, with no additional explanation.

**Important**: Output must follow the JSON structure specified below. No other content or fields are allowed.

---
**Original Question**:  
{question}
---

---
**AI Response (Context)**:  
{context}
---

---
{format_instructions}
---
""",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=AltQuestionsResponse).get_format_instructions()},
)    

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