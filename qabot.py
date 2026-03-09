from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

import gradio as gr
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM Setup
def get_llm():
    model_id = 'ibm/granite-3-8b-instruct'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.1,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm
    
## Task 1: Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## Task 2: Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Task 4: Vector database
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Task 3: Embedding model
def watsonx_embedding():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Task 5: Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']

# Gradio Interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="file"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Quest Analytics RAG Assistant",
    description="Upload a research paper and ask the AI assistant for real-time insights."
)

if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860, share=True)