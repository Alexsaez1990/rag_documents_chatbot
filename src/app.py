import streamlit as st
from streamlit_chat import message as st_message
import streamlit.components.v1 as components
from rag import RAGModel


@st.cache_resource
def initialize_ragmodel():
    """
    """
    return RAGModel()


class RAGApp:
    def __init__(self):
        self.ragmodel = initialize_ragmodel()
    
    def generate_answer(self):
        """
        """
        request = st.session_state.request
        with st.spinner("Generating response..."):
            response = self.ragmodel(request=request)
        
        st.session_state.history.append({"message": request, "is_user": True})
        st.session_state.history.append({"message": response, "is_user": False})
    
    def run_app(self):
        """
        """
        st.title("AI Chatbot Assistant for PDF documents")

        with st.container(border=True):
            if "history" not in st.session_state:
                st.session_state.history = []
            
            for i, chat in enumerate(reversed(st.session_state.history)):
                st_message(**chat, key=str((i))) # Unpack messages
        
        st.text_input("", key="request", on_change=self.generate_answer)
        if st.button("Clear"):
            st.session_state.history = []

if __name__ == '__main__':
    ragqa = RAGApp()
    ragqa.run_app()
