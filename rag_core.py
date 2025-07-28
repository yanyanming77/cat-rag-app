# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules["pysqlite3"]

# import pysqlite3
from sqlalchemy import create_engine, text, inspect
import sqlite3

import os
import re
from pathlib import Path
import pandas as pd

import streamlit as st
from bs4 import BeautifulSoup

from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import uuid
from openai import OpenAI
import openai

######################## DEFINE OPENAI API KEY #######################
openai.api_key = st.secrets['openai_api_key']
client = OpenAI(
  api_key=openai.api_key
)

######################## DEFINE EBMEDDING FUNCTION #######################
# Use openAI's embedding model
@st.cache_resource
def load_embedding_function(api_key):
    return OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=api_key)
embedding_function = load_embedding_function(openAI_api_key)

######################## LOAD CHROMA DB #######################
persist_dir = Path('./chroma_db')

@st.cache_resource
def load_chroma():
    vector_stores = {}
    for topic in ["Cats_Grooming_Tips", "Cat_Nutrition_Tips", "Cats_and_Babies", "Common_Cat_Behavior", "Common_Cat_Diseases"]:  
        vector_stores[topic] = Chroma(
            persist_directory=f"{persist_dir}/{topic}",
            embedding_function=embedding_function  # Use the same embedding function
        )
    print(f"✅ Loaded vector store for {topic} from ChromaDB.")
    return vector_stores

vector_stores = load_chroma()

######################## DEFINE LLM to use#######################
# define gpt-4o-mini as the llm used for topic classification
@st.cache_resource
def load_llm(api_key):
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0  # Low temperature ensures deterministic classification
    )
    return llm
llm = load_llm(client.api_key)

# define the llm for response generation
@st.cache_resource
def load_response_llm(api_key):
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.6
    )
llm_response = load_response_llm(client.api_key)

######################## DEFINE SQL DB CONNECTION #######################
# Initialize SQL Database Connection
@st.cache_resource
def get_db_engine():
    base_dir = os.path.dirname(__file__)  # folder where app.py lives
    db_path = os.path.join(base_dir, "cat_knowledge.db")
    db_uri = f"sqlite:///{db_path}"
    return create_engine(db_uri)

db_engine = get_db_engine()

######################## DEFINE FUNCTION #######################
######################## TEXT TO SQL #######################
def generate_sql_gpt4omini(query, print_query = False):
    """Generate SQL query using OpenAI GPT-4o-mini."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "Please convert natural language questions related to plant toxicity to cats into SQL queries.\n\n"
                "Please note, asking about plant **safety** is the same as asking about plant **toxicity**.\n"
                "When the user refers to a **plant in plural form** (e.g., 'roses', 'lilies'), treat it as the **singular plant name** ('rose', 'lily').\n"  # 💡 Added this line
                "Use basic singularization logic (e.g., remove plural 's', or handle common plurals like 'ies' → 'y') when necessary.\n\n"  # 💡 Added this line

                "The database name is `cat_knowledge.db`, and the table is called `plants`.\n\n"
                "The table has 4 columns:\n"
                "- `PlantName` (TEXT)\n"
                "- `ScientificName` (TEXT)\n"
                "- `Family` (TEXT)\n"
                "- `Toxicity` (TEXT, only has two values: Toxic, Non-Toxic)\n\n"

                "### Query Requirements:\n"
                "1. Use **case-insensitive search** for **all string comparisons**.\n"
                "2. Use **fuzzy search** (`LIKE '%' || query || '%'`) for matching **PlantName, ScientificName, and Family**.\n"
                "3. Use **case-insensitive search** for the **Toxicity** column.\n"
                "4. When querying about specific plants, search **both `PlantName` and `ScientificName`**.\n"
                "5. Do **not** use `ILIKE` (which is not supported in SQLite); instead, use:\n"
                "   ```sql\n"
                "   LOWER(column_name) LIKE LOWER('%query%')\n"
                "   ```"
            ),
        },
        {"role": "user", "content": f"Convert this to SQL: {query}"}
    ],
    temperature=0,  # Use 0 for deterministic responses
    max_tokens=200
)

    sql_query = response.choices[0].message.content
    # remove markdown formatting 
    sql_query = re.sub(r"```sql\s*", "", sql_query)  # Remove opening triple backticks
    sql_query = re.sub(r"```", "", sql_query)  # Remove closing triple backticks
    sql_query = sql_query.strip()

    if print_query:
        print(sql_query)

    return sql_query

######################## DEFINE FUNCTION #######################
######################## EXECUTE SQL #######################
def execute_sql(sql_query):
    """Execute SQL query on the database."""
    with db_engine.connect() as connection:
        try:
            result = connection.execute(text(sql_query))
            return result.fetchall()
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
        
######################## TOPIC CLASSIFICATION #######################
classification_prompt = PromptTemplate.from_template("""
Classify the following query into one or more of these categories:
1. Cats Grooming
2. Cat Nutrition
3. Cats and Babies
4. Common Cat Behavior
5. Common Cat Diseases
6. Cats Plant Toxicity
7. None

- If the question is about how cats live, their behavior, or lifespan, include "Common Cat Behavior".
- If the question involves illnesses, aging-related issues, or health risks, include "Common Cat Diseases".
- If the question involves plant safety for cats, include "Cats Plant Toxicity".
- If multiple categories apply, list all relevant ones.
- If the question is not related to any of 1-6, then return 'None'

Query: {query}
Categories (comma-separated):
""")

# Create LLM chain for classification
# Allow for multiple topics
classification_chain = LLMChain(llm=llm, prompt=classification_prompt)

######################## EXTRACT TOXICITY-RELATED PARTS FOR SQL QUERY #######################
# Define a prompt to extract the toxicity-related part from the query
extract_toxicity_prompt = PromptTemplate.from_template("""
Please extract the part that asks about plant toxicity to cats from the query.

Query: {query}

If a specific plant is mentioned in the query, replace the plant name in the extracted question with a list of all its known names (scientific name, common synonyms, botanical and regional names), separated by commas.

Ensure the alternative names are inserted **inline** in place of the original plant name, not in parentheses.

Return only the extracted question without adding any extra words.
""")

extract_toxicity_chain = LLMChain(llm=llm, prompt=extract_toxicity_prompt)

######################## DEFINE FUNCTION #######################
######################## RETRIEVE DOCUMENT AND RUN SQL QUERY #######################
def retrieve_documents_and_sql(query):
    """
    Classify the query, retrieve from the right collection, and return results.
    integrate the sql-query here too after the topic classification
    """
    predicted_topic = classification_chain.run(query)
    # convert the result to a list 
    predicted_topic = [cat.strip() for cat in predicted_topic.split(",")]
    print(f'Predicted topic: {predicted_topic}')
    
    if predicted_topic != 'None':
        # Map classification output to stored ChromaDB collection names
        topic_mapping = {
            "Cats Grooming": "Cats_Grooming_Tips",
            "Cat Nutrition": "Cat_Nutrition_Tips",
            "Cats and Babies": "Cats_and_Babies",
            "Common Cat Behavior": "Common_Cat_Behavior",
            "Common Cat Diseases": "Common_Cat_Diseases"
        }
        
        retrieved_results = {} # the dictionary that stores 'general' and 'sql'
        retrieved_results['general'] = []
        retrieved_results['sql'] = []

        retrieved_docs = []
        # iteratve over each predicted topic
        # use multi-query approach (use llm to rephrase multiple versions of the prompt)
        for topic in predicted_topic:
            if topic in topic_mapping.keys():
                collection_name = topic_mapping[topic]
                vector_db = vector_stores[collection_name]
                # Perform search
                multi_query_prompt = PromptTemplate.from_template(
                    """Generate 5 diverse search queries for this question: {question} \n
                    Keep the result to only contain the 5 generated questions as list of strings.
                    """
                )
                doc_retriever = vector_db.as_retriever(search_kwargs={"k":8})
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever = doc_retriever,
                    llm = llm_response,
                    prompt = multi_query_prompt
                )
                # get documents
                docs = multi_query_retriever.get_relevant_documents(query)
                multi_queries = multi_query_retriever.llm_chain.invoke({"question": query})
                # print(f"generated sub-queries: \n{multi_queries}")
                retrieved_docs.extend(docs)
            
        # deduplicate documents if needed
        unique_docs = {doc.page_content: doc for doc in retrieved_docs}.values()
        retrieved_docs = list(unique_docs) 
        # print(f"retrieved documents: \n{retrieved_docs}")
        retrieved_results['general'] = retrieved_docs

        # if Cats Plan Toxicity is in the topic and sql query needs to be ran
        if 'Cats Plant Toxicity' in predicted_topic:
            if len(predicted_topic) > 1:
                # extract the toxicity part from the query
                toxicity_part_question = extract_toxicity_chain.run(query)
            elif len(predicted_topic) == 1:
                toxicity_part_question = query
            # print(toxicity_part_question)
            
            sql_query = generate_sql_gpt4omini(toxicity_part_question, True)
            sql_query_result = execute_sql(sql_query)
            retrieved_results['sql'] = sql_query_result
        else:
            toxicity_part_question = ''
            sql_query = ''
    
    # if the topic is not related to documents stored in chroma_db
    else:
        toxicity_part_question = ''
        sql_query = ''
        retrieved_results = None
    # return the toxicity_part_question, sql_query, and the final result
    return toxicity_part_question, sql_query, retrieved_results


######################## GENERATE ANSWER BASED ON RETRIEVED DOCUMENTS AND SQL RESULTS #######################
# Define the prompt template
prompt_template_response = PromptTemplate.from_template("""
You are a cat expert and will answer user's questions based on the following contexts:
retrieved document, the SQL database, the conversation history, and the user uploaded document

Respond in a conversational tone — helpful, relaxed, and easy to understand, like you're talking to a fellow pet owner, you could use emojis if necessary.
The respond should be short and concise.

- If the `toxicity_part_question` and `toxicity_sql_query` are empty, provide the answer based only on `document_context`.
- If `toxicity_part_question` and `toxicity_sql_query` are **not empty**, use both `sql_context` and `document_context` to generate the answer.
- If the `sql_context` contains a **list of tuples in the form of** (plantname, plantname, toxicity) **or** (plantname, toxicity), **aggregate the results carefully**:
  - **List all plants that are toxic separately from those that are non-toxic.**
  - **If a plant has both toxic and non-toxic varieties, mention them explicitly.**
  - **Do NOT assume all plants in a group are toxic if at least one is non-toxic.**

If you do not find enough information in the retrieved knowledge, say:  
*"Sorry, I don't find enough information from the database. Please try another question."*

---

### **Retrieved Knowledge**
{document_context}

### **SQL Context**
{sql_context}

### **Toxicity SQL Query**
{toxicity_sql_query}

---

### **User Query**
{query}

### **Conversation History**
{user_doc_context}

### **Retrieved knowledge from user uploads**
{history_context}

### **Answer**
""")

# Create LLM chain
llm_chain_response = LLMChain(llm=llm_response, prompt=prompt_template_response)

# Define a function to generate answer
# add short-term memory
def generate_answer(query, history = None, user_doc = None, print_details = False):
    # retrieve
    toxicity_part_question, toxicity_sql_query, retrieved_results = retrieve_documents_and_sql(query)
    
    if toxicity_part_question != '' and toxicity_sql_query!= '':
        if print_details:
            print(f"toxicity part question:\n{toxicity_part_question}\n")
            print(f"toxicity part sql query:\n{toxicity_sql_query}\n")
            print(f"retrieved sql results:\n{retrieved_results['sql']}")

    # create answer
    if retrieved_results:
        document_context = "\n\n".join([doc.page_content for doc in retrieved_results['general']])
    else:
        document_context = ''
    # if there is history
    if history:
        history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    else:
        history_context = ''
    
    # if there is user uploaded document
    if user_doc:
        user_doc_context = "\n\n".join([doc.page_content for doc in user_doc])
    else: 
        user_doc_context = ''

    sql_context = retrieved_results['sql']
    answer = llm_chain_response.run({"document_context": document_context, \
                                    "history_context": history_context, \
                                    "user_doc_context": user_doc_context, \
                                    "sql_context":sql_context, \
                                    "toxicity_sql_query": toxicity_sql_query, "query": query})
    return answer


######################## OTHER FUNCTIONS #######################
def strip_html(text):
    """Prettify html text"""
    return BeautifulSoup(text, "html.parser").get_text().strip()

def load_and_chunk_document(file_path):
    """load user-uploaded document"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    
    # chunk and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = text_splitter.split_documents(loader.load())

    return chunks

def create_or_update_faiss(faiss_store, doc_chunks):
    # Convert Document objects to plain strings
    texts = [doc.page_content for doc in doc_chunks]

    # Either create or extend FAISS store
    if faiss_store is None:
        # First doc: create a new FAISS store
        return FAISS.from_texts(texts, embedding_function)
    else:
        # Add to existing FAISS store
        faiss_store.add_texts(texts)
        return faiss_store

def retrieve_from_upload_document(query):
    """retrieve from the user uploaded vector db"""

    result = st.session_state.faiss_store.similarity_search(query, k=3)

    return result