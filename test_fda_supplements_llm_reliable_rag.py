# -*- coding: utf-8 -*-
# # %%
# Install required packages
#!pip install langchain langchain-community python-dotenv
#!pip install langchain-cohere tiktoken chromadb
# %%
#!pip install langchain-openai
# %%
### LLMs
import os
from dotenv import load_dotenv

# Load environment variables from '.env' file
load_dotenv()


# %%
"""### Create Vectorstore"""

### Build Index
import langchain
import faiss
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


os.environ['USER_AGENT'] = 'myagent'

# Set embeddings
embedding_model = OpenAIEmbeddings()

import pandas as pd
data = pd.read_csv('/Users/catherinez/VSC/athena-ai/data/fda_dietary_supplement_warning_letters_with_text.csv')
data['Letter Text'] = data['Letter Text'].str.replace(
    r"WARNING LETTER\s*More Warning Letters\s*Warning LettersAbout Warning and Close-Out Letters",
    "",
    regex=True
)
data['Letter Text'] = data['Letter Text'].str.replace(r"\nAbout Warning and Close-Out Letters\n", "", regex=True)
# %%
data.to_csv('/Users/catherinez/VSC/athena-ai/data/fda_dietary_supplement_warning_letters_with_text_cleaned.csv', index=False)  
# %%
# don't use
from fpdf import FPDF

# Initialize the PDF object
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Add a Unicode-compatible font
pdf.add_font("DejaVu", style="", fname="/Users/catherinez/Downloads/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf", uni=True)
pdf.set_font("DejaVu", size=12)

# Iterate through the Letter Text column and add each entry to the PDF
for i, letter in enumerate(data['Letter Text']):
    pdf.add_page()
    pdf.multi_cell(0, 10, f"Letter {i+1}\n\n{letter}")

# Save the PDF
output_path = "/Users/catherinez/VSC/athena-ai/data/fda_letters.pdf"
pdf.output(output_path)
# %%
fda_letters = data['Letter Text'].tolist()
fda_letters = fda_letters[:80]

docs_list = [Document(page_content=letter) for letter in fda_letters] # Create Document objects
# %%
# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
# %%
# don't run
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

# Calculate total tokens
total_tokens = sum(len(tokenizer.encode(doc.page_content)) for doc in doc_splits)
print(f"Total tokens: {total_tokens}")
for i, doc in enumerate(doc_splits):
    tokens = tokenizer.encode(doc.page_content)
    print(f"Document {i+1} token count: {len(tokens)}")
    print(f"Document {i+1} tokens: {tokens[:20]}...")  # Print the first 20 tokens for brevity
    print(f"Document {i+1} content preview: {doc.page_content[:100]}...\n")
# sum tokens

# %%
 # Print the first 100 characters of the document
# %%
# Add to vectorstore
'''
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag",
    embedding=embedding_model,
)
'''
index = faiss.IndexFlatL2(1536)

vectorstore = FAISS(
    embedding_function=embedding_model,
    index=index,
    index_to_docstore_id={},
    docstore=InMemoryDocstore(),
)

uuids = [str(uuid4()) for _ in range(len(doc_splits))]
vectorstore.add_documents(doc_splits, ids=uuids)

retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 10}, # number of documents to retrieve
            )
# %%
"""### Question"""

question = "Is it risky if you say on your label the following: - You will get bigger and stronger. What law is relevant?"

"""### Retrieve docs"""

docs = retriever.invoke(question)

"""### Check document relevancy"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
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

"""### Filter out the non-relevant docs"""

docs_to_use = []
for doc in docs:
    print(doc.page_content, '\n', '-'*50)
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res,'\n')
    if res.binary_score == 'yes':
        docs_to_use.append(doc)

"""### Generate Result"""

from langchain_core.output_parsers import StrOutputParser

# Prompt
system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge.
Use three-to-five sentences maximum and keep the answer concise."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
    ]
)

# LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n".join(
        f"<doc{i+1}>:\n"
        f"Content: {doc.page_content}\n"
        f"</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )
# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"documents":format_docs(docs_to_use), "question": question})
print(generation)
# %%
"""### Check for Hallucinations"""

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in 'generation' answer."""

    binary_score: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# LLM with function call
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

response = hallucination_grader.invoke({"documents": format_docs(docs_to_use), "generation": generation})
print(response)
# %%

"""### Highlight used docs"""

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Data model
class HighlightDocuments(BaseModel):
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

# LLM
llm = ChatGroq(model="mistral-saba-24b", temperature=0)

# parser
parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

# Prompt
system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
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


prompt = PromptTemplate(
    template= system,
    input_variables=["documents", "question", "generation"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Chain
doc_lookup = prompt | llm | parser

# Run
lookup_response = doc_lookup.invoke({"documents":format_docs(docs_to_use), "question": question, "generation": generation})

for id, title, source, segment in zip(lookup_response.id, lookup_response.title, lookup_response.source, lookup_response.segment):
    print(f"ID: {id}\nTitle: {title}\nSource: {source}\nText Segment: {segment}\n")

# %%
