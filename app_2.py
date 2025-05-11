import os
import streamlit as st
import pandas as pd
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

# Load API keys
os.environ['USER_AGENT'] = 'myagent'
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


def display_document_with_laws(doc, doc_laws, idx):
    st.markdown(f"### Document {idx + 1}")
    st.markdown(f"[🔗 View Full Letter]({doc.metadata.get('source', 'Unknown')})")
    st.write(doc.page_content)

    if doc_laws:
        st.markdown("**Cited Laws, Rules, or Codes in This Document:**")
        for law in set(doc_laws):
            st.markdown(f"- {law}")
    else:
        st.markdown("")


st.title("⚖️ Supplement Claim Compliance Checker")
st.write("Analyze your claim for FDA regulatory risk and explore supporting evidence from official warning letters.")

user_claim = st.text_area("📝 **Enter your claim or statement:**", height=150)

if st.button("🚀 Analyze Claim"):
    with st.spinner("🔍 Analyzing your claim, retrieving evidence, and extracting legal references..."):
        with st.status("⚙️ Preparing analysis...") as status:
            st.write("🔄 Loading regulatory data...")
            data = pd.read_csv('data/fda_dietary_supplement_warning_letters_with_text_cleaned.csv')
            fda_letters = data['Letter Text'].tolist()[:80]
            urls = data['URL'].tolist()[:80]
            status.update(label="✅ Data loaded successfully.")

            docs_list = [Document(page_content=letter, metadata={"source": urls[i]}) for i, letter in enumerate(fda_letters)]

            st.write("🔄 Splitting and embedding documents...")
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
            doc_splits = text_splitter.split_documents(docs_list)
            embedding_model = OpenAIEmbeddings()
            index = faiss.IndexFlatL2(1536)
            vectorstore = FAISS(embedding_function=embedding_model, index=index, index_to_docstore_id={}, docstore=InMemoryDocstore())
            uuids = [str(uuid4()) for _ in range(len(doc_splits))]
            vectorstore.add_documents(doc_splits, ids=uuids)
            status.update(label="✅ Documents processed and embedded.")

            st.write("🔍 Retrieving documents related to your claim...")
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
            search_query = f"Is this claim likely to trigger FDA enforcement? {user_claim}"
            retrieved_docs = retriever.invoke(search_query)
            status.update(label="✅ Retrieved top relevant documents.")

            st.write("🔍 Scoring document relevance...")
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

            graded_docs = []
            for doc in retrieved_docs:
                # print(doc.page_content, '\n', '-'*50)
                result = retrieval_grader.invoke({"question": search_query, "document": doc.page_content})
                # print(res,'\n')
                if result.binary_score == 'yes':
                    graded_docs.append({"doc": doc, "is_relevant": result.binary_score.lower() == 'yes'})

            status.update(label="✅ Documents graded for relevance.")

            st.write("📝 Generating regulatory risk assessment...")
            def format_docs(docs):
                return "\n".join(f"<doc{i+1}>:\nContent: {doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate([d['doc'] for d in docs]))

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "Provide a regulatory risk assessment based on the provided documents. Be specific and concise. Include multiple types of regulatory risks that are relevant to the claim."),
                ("human", "Documents:\n{documents}\n\nQuestion:\n{question}")
            ])
            answer_chain = answer_prompt | llm | StrOutputParser()
            assessment = answer_chain.invoke({"documents": format_docs(graded_docs), "question": search_query})

            st.write("🔎 Extracting supporting snippets and legal references...")
            class HighlightWithLaws(BaseModel):
                """Return the specific part of a document used for answering the question."""

                id: List[str] = Field(
                    ...,
                    description="List of id of docs used to answers the question"
                )

                title: List[str] = Field(
                    ...,
                    description="List of titles used to answers the question"
                )

                source: List[str] = Field(
                    ...,
                    description="List of sources used to answers the question"
                )

                segment: List[str] = Field(
                    ...,
                    description="List of direct segements from used documents that answers the question"
                )

                laws: List[str] = Field(
                    ...,
                    description="List of laws from used documents"
                )

                source: List[str] = Field(
                    ...,
                    description="List of sources from used documents"
                )

            combined_parser = PydanticOutputParser(pydantic_object=HighlightWithLaws)
            combined_prompt_template = """You are an advanced assistant for document search and retrieval. You are provided with the following:
            1. A question.
            2. A generated answer based on the question.
            3. A set of documents that were referenced in generating the answer.

            Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to
            generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text
            in the provided documents. You must also list all specific laws, rules, or codes mentioned in each snippet. You must also provide examples of
            how to improve the claim based on the laws, rules, or codes mentioned in the documents.

            Ensure that:
            - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
            - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
            - (Important) If you didn't used the specific document don't mention it.

            Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

            <format_instruction>
            {format_instructions}
            </format_instruction>
            """
            
            combined_prompt = PromptTemplate(
                template=combined_prompt_template,
                input_variables=["documents", "question", "generation"],
                partial_variables={"format_instructions": combined_parser.get_format_instructions()}
            )
            combined_chain = combined_prompt | ChatGroq(model="llama-3.3-70b-versatile", temperature=0) | combined_parser
            combined_response = combined_chain.invoke({"documents": format_docs(graded_docs), "question": search_query, "generation": assessment})

            status.update(label="✅ Analysis complete.", state="complete")

        # Display Assessment
        st.markdown("## ✅ **Regulatory Risk Assessment**")
        st.write(assessment)

        # Show Snippets with Laws for Reference
        st.markdown("## 📄 **Snippets of FDA Warning Letters with Laws**")
        for idx, graded in enumerate(graded_docs):
            doc = graded['doc']

            st.markdown(f"### Document {idx + 1}")
            st.markdown(f"[🔗 View Full Letter]({doc.metadata.get('source', 'Unknown')})")
            st.write(doc.page_content)
