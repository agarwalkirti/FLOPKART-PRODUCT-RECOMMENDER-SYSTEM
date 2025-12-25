from langchain_groq import ChatGroq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

from flopkart.config import Config


class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(
            model=Config.RAG_MODEL,
            temperature=0.5,
        )
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        # Retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Prompt to convert follow-ups into standalone questions
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Prompt for answering using retrieved context
        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an e-commerce assistant. Answer product-related questions "
                "using only the provided context. Be concise and helpful.\n\n"
                "CONTEXT:\n{context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # History-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=self.model,
            retriever=retriever,
            prompt=contextualize_prompt,
        )

        # QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=qa_prompt,
        )

        # RAG chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain,
        )

        # Attach message history
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
