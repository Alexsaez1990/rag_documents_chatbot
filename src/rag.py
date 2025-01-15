import getpass
import os
import configparser
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser, Document
from langchain.prompts import PromptTemplate
from transformers import GPT2Tokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


config = configparser.ConfigParser()
config.read('config.ini')
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['API_KEYS']['huggingface_token']


class RAGModel():
    """
    """
    def __init__(self):
        self.file_documents_path = "./data/pdf_documents"
        self.path_vector_store = "./vector_db"
        self.cache_models_path = "./cache_models"
        self.model_name = "google/flan-t5-large"
        #self.model_name = "EleutherAI/gpt-neo-1.3B"
        #self.model_name = "EleutherAI/gpt-neo-125M"
        self.embeddings = HuggingFaceEmbeddings( # NOT USE GPT MODEL FOR EMBEDDINGS
            model_name="google/flan-t5-large",
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True}
        ) 
        self.load_data()
        self.split_document()
        self.load_vector_store()
        self.retrieve()
        

    def __call__(self, request:str):
        return self.generate(request=request)
    
    def load_data(self):
        """
        """
        #loader = PyMuPDFLoader(file_path=self.file_documents_path)
        if not os.path.exists(self.file_documents_path):
            # TODO: MAKE EXCEPTION ERROR
            print(f"Directory {self.file_documents_path} does not exists")
        loader=PyPDFDirectoryLoader(self.file_documents_path)
        self.docs = loader.load()
        '''self.docs = [
            Document(page_content=clean_text(doc.page_content), metadata=doc.metadata) for doc in self.docs
        ]'''
        print(f"Loaded and clean {len(self.docs)} documents.")
    
    def split_document(self):
        """
        """
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
        add_start_index=True
        )

        self.splits = text_splitter.split_documents(self.docs)
    
    def vector_store(self):
        """
        """
        # CHECK USING CHROMA
        self.vector_store = FAISS.from_documents(documents=self.splits, embedding=self.embeddings)
        self.save_vector_store()
    
    def save_vector_store(self):
        """
        """
        if not os.path.exists(self.path_vector_store):
            os.makedirs(self.path_vector_store)
        self.vector_store.save_local(self.path_vector_store)
    
    def load_vector_store(self):
        """
        """
        index_file = os.path.join(self.path_vector_store, "index.faiss")
        if os.path.exists(index_file):
            print("Loading existing FAISS index...")
            self.vector_store = FAISS.load_local(self.path_vector_store, embeddings=self.embeddings, allow_dangerous_deserialization=True)
        else:
            print(f"FAISS index not found. Creating a new one...")
            self.vector_store()
            self.save_vector_store()
    
    def retrieve(self):
        """
        """
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    def generate(self, request:str):
        """
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_models_path)
        llm = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=self.cache_models_path)

        retrieved_docs = self.retriever.get_relevant_documents(request)
        retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
        print(f"RETRIEVED CONTEXT: {retrieved_context}")
        def format_docs(docs):
            return "n\n".join(doc.page_content for doc in docs)
        
        chunks = [retrieved_context[i: i + 512] for i in range(0, len(retrieved_context), 512)]

        response = ""
        for chunk in chunks:
            objective_prompt = PromptTemplate.from_template(
                f""" 
                Contexto del documento:
                {chunk}

                Pregunta: {request}

                Respuesta:
                """
            ).format(context=retrieved_context, question=request)

            

            inputs = tokenizer(objective_prompt, return_tensors="pt", max_length=1024, truncation=True)
            outputs = llm.generate(**inputs, max_length=512, temperature=0.7, do_sample=True)
            chunk_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            response += chunk_response + "\n"
        print(f"RESPONSE: {response}")
        try:
            #response = "".join(chunk for chunk in rag_chain.stream(request))
            return self.clean_response(response)
        except Exception as e:
            print(f"Error during generation {e}")
            raise
        
    def clean_response(self, response):
        """
        """
        lines = response.split("\n")
        seen = set()
        cleaned_lines = []
        for line in lines:
            clean_line = line.strip()
            if clean_line and clean_line not in seen:
                seen.add(clean_line)
                cleaned_lines.append(clean_line)
        
        return "\n".join(cleaned_lines)
    
    def retrieve_model(self):
        hf_hub_download(self.model_name, filename="pytorch_model.bin", cache_dir=self.cache_models_path)
    
import re
def clean_text(text):
    text = re.sub(f"\\n+", " ", text) # Remove excessive line breaks
    text = re.sub(f"\\s+", " ", text) # Normalize white spaces
    text = re.sub(r"Page \d+", "", text) # Remove page numbers
    return text.strip()
        


# Cosas pendientes:
# - Ver que diga cosas con más sentido. La salida es sucia y se ve el prompt y todo. Se debería poder limpiar eso
# - Guardar el modelo en local -> Importante para poder trabajar con modelos grandes
# - Ver cómo limitar el tamaño de la respuesta
# - Probar otros documentos
# - Probar tamaños splitter