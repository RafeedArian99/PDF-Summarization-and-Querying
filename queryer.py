from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tqdm import tqdm
import time

if not load_dotenv():
    raise FileNotFoundError('".env" file not found')


class Queryer:
    """
    Create a new Queryer that will initialize embeddings and LLM
    """

    def __init__(self) -> None:
        print(end="Initializing... ", flush=True)
        self.embeddings = HuggingFaceEmbeddings()
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7)
        self.chain = None
        self.chat_history = []
        print("DONE")

    def process_file(self, file_path: str) -> None:
        """
        Pass the path to a PDF file for the Queryer to process.
        It may take some time for large PDF's.
        """
        raw_text = ""
        pdfreader = PdfReader(file_path)

        for page in tqdm(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", "."],
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        print(end="Creating vector store... ", flush=True)
        vector_store = FAISS.from_texts(texts, self.embeddings)
        print("DONE!")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, chain_type="stuff", retriever=retriever
        )

    def ask(self, query: str) -> str:
        """
        Pass a question for the Queryer. Returns the LLM's response.
        """
        if not query or query.isspace():
            return None

        # query += "\n\nPlease refrain from repeating yourself."
        start_time = time.time()
        print(end="Generating response...", flush=True)
        response = self.chain.invoke({"question": query, "chat_history": self.chat_history})
        response = response["answer"].strip()
        self.chat_history.append((query, response))
        print(f"DONE ({time.time() - start_time}s)")
        return response
