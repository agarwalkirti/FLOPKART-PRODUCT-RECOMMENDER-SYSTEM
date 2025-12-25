from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from flopkart.config import Config


class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(
            model=Config.RAG_MODEL,
            temperature=0.5
        )
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Rewrite question using history
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user's question as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        contextualize_chain = (
            contextualize_prompt
            | self.model
            | StrOutputParser()
        )

        # Retrieve documents
        retrieve_chain = RunnableLambda(
            lambda x: retriever.invoke(x["standalone_question"])
        )

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an e-commerce assistant. Answer strictly using the context.\n\nCONTEXT:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        qa_chain = (
            qa_prompt
            | self.model
            | StrOutputParser()
        )

        rag_chain = (
            {
                "standalone_question": contextualize_chain,
                "input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
            | RunnablePassthrough.assign(
                context=lambda x: retrieve_chain.invoke(x)
            )
            | RunnablePassthrough.assign(
                answer=qa_chain
            )
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
