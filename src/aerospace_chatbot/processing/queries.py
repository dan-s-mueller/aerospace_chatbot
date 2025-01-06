"""QA model and retrieval logic."""

import logging

# Utilities
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph, START, END

# Typing
from typing_extensions import List
from typing import List, Literal, Tuple

# Services
from aerospace_chatbot.services.prompts import (InLineCitationsResponse, 
                                                AltQuestionsResponse,
                                                style_mode, 
                                                CHATBOT_SYSTEM_PROMPT, 
                                                QA_PROMPT, 
                                                SUMMARIZE_TEXT, 
                                                GENERATE_SIMILAR_QUESTIONS_W_CONTEXT
                                                )

class QAModel:
    """
    Handles question answering and retrieval.
    """

    def __init__(self,
                 db_service,
                 llm_service,
                 k_retrieve=20,
                 k_rerank=5,
                 style=None,
                 memory_config=None):
        """
        Initialize QA model with necessary services.
        """
        self.db_service = db_service
        self.llm_service = llm_service
        self.k_retrieve = k_retrieve
        self.k_rerank = k_rerank
        self.style = style   # Validated when style_mode is called
        self.workflow = None
        self.result = None
        self.memory_config = memory_config
        self.logger = logging.getLogger(__name__)
        
        # Get retrievers from database services
        self.db_service.get_retriever(k=self.k_retrieve)

        # Compile workflow
        if self.memory_config is None: 
            # Set to default memory config
            self.memory_config = {"configurable": {"thread_id": "1"}}
        self.workflow = self._compile_workflow()
        
    def query(self,query): 
        """
        Executes a query and retrieves the relevant documents.
        """

        # Add answer to response, create an array as more prompts come in
        self.logger.info(f'Running query through workflow: {query}')
        self.result = self.workflow.invoke(
            {"messages": [("human", query)]}, 
            self.memory_config
        )
        # if not hasattr(self, 'result') or self.result is None:
        #     self.result = [answer_result]
        # else:
        #     self.result.append(answer_result)

        # # Add sources to response, create an array as more prompts come in
        # answer_sources = [data.metadata for data in self.result[-1]['references']]
        # answer_scores =  self.result[-1]['scores']
        # if not hasattr(self, 'sources') or self.sources is None:
        #     self.sources = [answer_sources]
        #     self.scores = [answer_scores]
        # else:
        #     self.sources.append(answer_sources)
        #     self.scores.append(answer_scores)

        # Add answer to memory
        # self.ai_response = self.result[-1]['answer'].content
        # self.memory.save_context({'question': query}, {'answer': self.ai_response})

    class State(MessagesState):
        """
        State class for the QA model.
        """
        context: List[Tuple[Document, float, float]]
        cited_sources: List[Tuple[Document, float, float]]
        alternative_questions: List[str]
        summary: str

    def _retrieve(self, state: State):
        """
        Retrieve the documents from the database.
        """
        self.logger.info(f"Node: retrieve")

        # Retrieve docs
        retrieved_docs = self.db_service.retriever.invoke(state["messages"][-1].content)
        self.logger.info(f"Retrieved docs")

        # Rerank docs
        reranked_docs = self.db_service.rerank(
            state["messages"][-1].content, 
            retrieved_docs, 
            top_n=self.k_rerank
        )
        self.logger.info(f"Reranked docs")

        return {"context": reranked_docs}

    def _generate_w_context(self, state: State):
        """
        Call the model with the prompt with context.
        """
        self.logger.info(f"Node: generate_w_context")

        # Get the summary, add system prompt
        summary = state.get("summary", "")
        system_prompt = CHATBOT_SYSTEM_PROMPT.format(style_mode=style_mode(self.style))
        # self.logger.info(f" generate_w_context system prompt: {system_prompt.content}")
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [system_prompt] + [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = [system_prompt] + state["messages"]

        # Add context to the prompt
        docs_content=""
        for i, (doc, _, rerank_score) in enumerate(state["context"]):
            # Source IDs in the order they show in in the array. Indexed from 1, retrieve with 0 index.
            if rerank_score is not None:    # Only include docs with a rerank score
                docs_content += f"Source ID: {i+1}\n{doc.page_content}\n\n"

        # Prompt with context and pydantic output parser
        prompt_with_context = QA_PROMPT.format(
            context=docs_content,
            question=state["messages"][-1].content, 
        )
        # Replace the last message (user question) with the prompt with context, return LLM response
        messages[-1] = prompt_with_context 
        response = self.llm_service.get_llm().invoke(messages)
        self.logger.info('Response generated')

        # Parse the response. This will return a InLineCitationsResponse object. 
        # This object has two fields: content and citations.
        # Replace the last message with the content of the parsed and validated response. 
        # AIMessage metadata will be incorrect.
        parsed_response = PydanticOutputParser(pydantic_object=InLineCitationsResponse).parse(response.content)
        response.content = parsed_response.content
        self.logger.info('Response parsed')

        # Return cited_sources as the list of tuples that matched the citations.
        existing_cited_sources = state.get("cited_sources", [])  # Grab whatever might already be in cited_sources
        cited_sources = [state["context"][int(citation)-1] for citation in parsed_response.citations]
        existing_cited_sources.append(cited_sources)  # Append the new list as a sublist
        state["cited_sources"] = existing_cited_sources

        # Update the state messages with the messages updated in this node.
        state["messages"] = messages
        return {"messages": [response], 
                "cited_sources": state["cited_sources"]}

    def _should_continue(self, state: State) -> Literal["summarize_conversation", END]:
        """
        Define the logic for determining whether to end or summarize the conversation
        """
        self.logger.info(f"Node: should_continue")

        # If there are more than six messages, then we summarize the conversation
        messages = state["messages"]
        if len(messages) > 6:
            self.logger.info(f"Summarizing conversation")
            return "summarize_conversation"
        
        # Otherwise just end
        self.logger.info(f"Ending conversation")
        return END

    def _summarize_conversation(self, state: State):
        """
        Summarize the conversation
        """
        self.logger.info(f"Node: summarize_conversation")

        summary = state.get("summary", "")
        if summary:
            # If a summary already exists, extend it
            summary_message = SUMMARIZE_TEXT.format(
                summary=summary,
                augment="Extend the summary provided by taking into account the new messages above."
            )
        else:
            # If no summary exists, create one
            summary_text="""---\n**Conversation Summary to Date**:\n{summary}\n---"""
            summary_message = SUMMARIZE_TEXT.format(
                summary=summary_text,
                augment="Create a summary of the conversation above."
            )

        messages = state["messages"] + [summary_message]
        response = self.llm_service.get_llm().invoke(messages)

        # Prune messages. This deletes all but the last two messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}
    
    def _generate_alternative_questions(self, state: State):
        """
        Generates alternative questions based on the prompt provided.
        """
        self.logger.info(f"Node: generate_alternative_questions")

        prompt = GENERATE_SIMILAR_QUESTIONS_W_CONTEXT.format(
            question=state["messages"][-2].content,  # Last user message is 2 messages back
            context=state["messages"][-1].content    # Last AI message is 1 message back
        )
        response = self.llm_service.get_llm().invoke(prompt)
        self.logger.info('Alternative questions generated.')

        # Split the string into a list of questions, removing empty strings and stripping whitespace
        alternative_questions = PydanticOutputParser(pydantic_object=AltQuestionsResponse).parse(response.content)
        self.logger.info(f'Alternative questions parsed')

        # Update the state with the new alternative questions
        existing_alternative_questions = state.get("alternative_questions", [])  # Grab whatever might already be in cited_sources
        existing_alternative_questions.append(alternative_questions.questions)  # Append the new list as a sublist
        state["alternative_questions"] = existing_alternative_questions
        
        return {"alternative_questions": state["alternative_questions"]}
    
    def _compile_workflow(self):
        """
        Compile the workflow.
        """
        # Compile application and test
        workflow = StateGraph(QAModel.State)

        # Define nodes
        workflow.add_node("retrieve", self._retrieve) 
        workflow.add_node("generate_w_context", self._generate_w_context)
        workflow.add_node("summarize_conversation", self._summarize_conversation)
        workflow.add_node("generate_alternative_questions", self._generate_alternative_questions)
        # Define edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate_w_context")
        workflow.add_edge("generate_w_context", "generate_alternative_questions")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            "generate_alternative_questions",     # Define the start node. We use `generate_alternative_questions`. This means these are the edges taken after the `conversation` node is called.
            self._should_continue,                # Next, pass in the function that will determine which node is called next.
        )

        # Add a normal edge from `summarize_conversation` to END. This means that after `summarize_conversation` is called, we end.
        workflow.add_edge("summarize_conversation", END)

        # Compile the workflow
        app = workflow.compile(checkpointer=MemorySaver())
        return app