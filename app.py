import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from typing import List
import faiss
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

os.environ['USER_AGENT'] = 'myagent'

st.title("⚖️ FDA Compliance Checker")
st.write("Analyze your marketing claim for FDA regulatory risk and explore supporting evidence from official warning letters.")

user_input = st.text_area("📝 **Enter your marketing claim:**", height=150)
st.write("**Note:** This tool is for testing purposes only and does not constitute legal advice. Always consult a qualified attorney for legal matters.")

if st.button("🚀 Check Claim"):
    with st.spinner("🔍 Analyzing your claim, retrieving evidence, and extracting legal references..."):

        # --- Load and Clean Data ---
        data = pd.read_csv('data/fda_dietary_supplement_warning_letters_with_text_cleaned.csv')
        fda_letters = data['Letter Text'].tolist()[:25]
        docs_list = [Document(page_content=letter) for letter in fda_letters]

        # --- Split & Embed ---
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        embedding_model = OpenAIEmbeddings()
        index = faiss.IndexFlatL2(1536)
        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=index,
            index_to_docstore_id={},
            docstore=InMemoryDocstore(),
        )
        uuids = [str(uuid4()) for _ in range(len(doc_splits))]
        vectorstore.add_documents(doc_splits, ids=uuids)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})

        # --- Retrieve Documents ---
        user_input = "Is this claim likely to trigger FDA enforcement? " + user_input
        retrieved_docs = retriever.invoke(user_input)

        # --- Relevance Grader ---
        class GradeDocuments(BaseModel):
            binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
                
        retrieval_grader = grade_prompt | structured_llm_grader

        docs_to_use = []
        for doc in retrieved_docs:
            res = retrieval_grader.invoke({"question": user_input, "document": doc.page_content})
            if res.binary_score.lower() == 'yes':
                docs_to_use.append(doc)

        if not docs_to_use:
            st.warning("No relevant documents found. Try rephrasing your claim.")
            st.stop()

        # --- Answer Generation ---
        def format_docs(docs):
            return "\n".join(f"<doc{i+1}>:\nContent: {doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Answer the question based on provided documents. Keep the response under five sentences."),
            ("human", "Documents:\n{documents}\n\nQuestion:\n{question}")
        ])
        rag_chain = answer_prompt | llm | StrOutputParser()
        answer = rag_chain.invoke({"documents": format_docs(docs_to_use), "question": user_input})

        # --- Hallucination Check ---
        class GradeHallucinations(BaseModel):
            binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
        
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Documents:\n{documents}\n\nAnswer:\n{generation}")
        ])
        hallucination_chain = hallucination_prompt | llm.with_structured_output(GradeHallucinations)
        hallucination_result = hallucination_chain.invoke({"documents": format_docs(docs_to_use), "generation": answer})

        # --- Highlight Snippets ---
        class HighlightDocuments(BaseModel):
            id: List[str]
            title: List[str]
            source: List[str]
            segment: List[str]

        highlight_parser = PydanticOutputParser(pydantic_object=HighlightDocuments)
        highlight_system_prompt = system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
        1. A question.
        2. A generated answer based on the question.
        3. A set of documents that were referenced in generating the answer.

        Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to
        generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text
        in the provided documents.

        Ensure that:
        - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
        - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
        - (Important) If you didn't used the specific document don't mention it.

        Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

        <format_instruction>
        {format_instructions}
        </format_instruction>
        """
        highlight_prompt = PromptTemplate(
            template=highlight_system_prompt + "\n\nDocuments:\n{documents}\n\nQuestion:\n{question}\n\nAnswer:\n{generation}\n\n{format_instructions}",
            input_variables=["documents", "question", "generation"],
            partial_variables={"format_instructions": highlight_parser.get_format_instructions()}
        )
        highlight_chain = highlight_prompt | ChatGroq(model="llama-3.3-70b-versatile", temperature=0) | highlight_parser
        highlight_response = highlight_chain.invoke({"documents": format_docs(docs_to_use), "question": user_input, "generation": answer})

        # --- Display Results ---
        st.markdown("## ✅ Risk Assessment")
        st.write(answer)
        st.markdown("#### Is the Answer Grounded?")
        st.write(hallucination_result.binary_score)

        st.markdown("## 📄 Supporting Evidence Snippets")
        if highlight_response.segment:
            for idx, snippet in enumerate(highlight_response.segment):
                st.markdown(f"**Snippet {idx+1}:** {snippet}")
        else:
            st.write("No specific snippets could be identified.")