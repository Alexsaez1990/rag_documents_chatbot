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
from transformers import GPT2Tokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


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
        #self.model_name = "google/flan-t5-large"
        self.model_name = "EleutherAI/gpt-neo-1.3B"
        #self.model_name = "EleutherAI/gpt-neo-125M"
        self.embeddings = HuggingFaceEmbeddings( # NOT USE GPT MODEL FOR EMBEDDINGS
            model_name="google/flan-t5-large",
            model_kwargs={'device':'cuda'},
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
    
    def generate(self, request:str):
        """
        """
        llm = HuggingFaceHub(
            repo_id=self.model_name,
            model_kwargs={"temperature":0.1, "max_length":200}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        objective_prompt = PromptTemplate.from_template(
            """ 
            Eres un asistente que extrae ideas clave de documentos de negocios para ayudar a mejorar una empresa.
            Enfócate en identificar los puntos clave, las áreas relevantes para mejorar y las recomendaciones importantes.
            Responde de forma concisa en viñetas, destacando solo la información más relevante.
            Si no hay suficiente información, di: "No se encontraron ideas accionables.

            Contexto del documento:
            {context}

            Pregunta: {question}
            Respuesta:
            """
        )

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | objective_prompt
            | llm
            | StrOutputParser()
        )
        #print("RAG chain context:", {"context": self.retriever | format_docs, "question": RunnablePassthrough()})

        try:
            response = "".join(chunk for chunk in rag_chain.stream(request))
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