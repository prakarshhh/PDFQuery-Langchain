# LangChain components to use
from langchain_community.vectorstores import Cassandra

from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import os

# Support for dataset retrieval with Hugging Face
from datasets import load_dataset

# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:
import cassio
from PyPDF2 import PdfReader

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:FKFxqdtvqpUXcAtIYLsXEJOM:2b4ce8649b7ff79151384b8e11f9d69aa3c421ee26e8369c66d3bb67bce474fc" # enter the "AstraCS:..." string found in in your Token JSON file
ASTRA_DB_ID = "b4c91d3a-5f95-40cb-a174-149faf745615" # enter your Database ID
OPENAI_API_KEY = "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" # enter your OpenAI key


pdfreader = PdfReader('bs2023_24.pdf')

from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
        
print(raw_text)

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

lm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)
from langchain.text_splitter import CharacterTextSplitter
# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

print(texts[:50])


astra_vector_store.add_texts(texts[:50])

print("Inserted %i headlines." % len(texts[:50]))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    print("\nQUESTION: \"%s\"" % query_text)
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))