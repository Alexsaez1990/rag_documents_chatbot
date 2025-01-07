import getpass
import os
import configparser
#import torch
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser


config = configparser.ConfigParser()
config.read('config.ini')
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['API_KEYS']['huggingface_token']


class RAGModel():
    """
    """
    def __init__(self):
        self.file_documents_path = "./data/pdf_documents"
        self.path_vectore_store = "./vector_db"
        self.model_name = "gpt2"
        self.load_data()
        self.split_document()
        self.vector_store()
        self.retrieve()

    def __call__(self, request:str):
        return self.generate(request=request)
    
    def load_data(self):
        """
        """
        #loader = PyMuPDFLoader(file_path=self.file_documents_path)
        loader=PyPDFDirectoryLoader(self.file_documents_path)
        self.docs = loader.load()
    
    def split_document(self):
        """
        """
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
        )

        self.splits = text_splitter.split_documents(self.docs)
    
    def vector_store(self):
        """
        """
        embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device':'cuda'},
        encode_kwargs={'normalize_embeddings':True}
        )

        self.vector_store = FAISS.from_documents(documents=self.splits, embedding=embeddings, persist_directory=self.path_vector_store)
    
    def retrieve(self):
        """
        """
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    def generate(self, request:str):
        """
        """
        llm = HuggingFaceHub(
            repo_id=self.model_name,
            model_kwargs={"temperature":0.0, "max_length":100}
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = hub.pull("rlm/rag-prompt")
        print(
            prompt.invoke(
                {"context": "filler context", "question": "filler question"}
            ).to_string()
        )
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        for chunk in rag_chain.stream(request):
            print(chunk, end="", flush=True)
        return "".join(chunk for chunk in rag_chain.stream(request))
        
        


