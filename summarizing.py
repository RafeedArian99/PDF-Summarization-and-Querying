from PyPDF2 import PdfReader
from io import BytesIO
from rouge_score import rouge_scorer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import textwrap
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

map_template = """summarize this concisely:
{docs}
Summary:"""

reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:"""

class Summarizer:
    """
    Create a new summarizer
    possibly use this for the model:

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map="auto")

    """

    def __init__(self, model, tokenizer):


        self.text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        generation_config = GenerationConfig.from_pretrained(model.config.name_or_path)
        generation_config.max_new_tokens = 2048
        generation_config.temperature = 0.0001
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )

        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
        self.llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})




        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        map_prompt = PromptTemplate.from_template(map_template)
        self.map_chain = LLMChain(llm=self.llm, prompt=map_prompt)


        
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        self.reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        self.combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain, document_variable_name="docs"
        )

        self.reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=self.combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=self.combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=2048,
        )

        self.map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=self.map_chain,
            # Reduce chain
            reduce_documents_chain=self.reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )


    def file_to_docs(self, file):
        """Returns the file split into docs
        """
        loader = PyPDFLoader(file)
        docs = loader.load_and_split(self.text_splitter)
        return docs

    def run_chain(self, docs, chain):
        """Gets docs from file and runs the chain on docs
        """
        res = chain.run(docs)
        return res

    def run_map_chain(self, file):
        return self.run_chain(file, self.map_chain)

    def run_map_reduce_chain(self, file):
        return self.run_chain(file, self.map_reduce_chain)

    def run_reduce_chain(self, file):
        return self.run_chain(file, self.reduce_chain)

    def get_summary(self, x):
        x = x[x.find('Summary:') + len('Summary: '):]
        return x[:x.find('```')]

    def slow_summarize(self, file):
        """Summarize file (takes 13 mins for article)

        :return: summary of the file
        :rtype: str
        """
        res = list(map(lambda x: self.run_chain(x.page_content, self.map_chain), self.file_to_docs(file)))
        res = list(map(lambda x: self.get_summary(x), res))
        res = "\n".join(res)
        return textwrap.fill(res, width=100)

    def summarize(self, file):
        """Summarize file

        :return: summary of the file
        :rtype: str
        """
        res = self.map_chain.run(self.file_to_docs(file))
        return textwrap.fill(res, width=100)



# class Queryer:
#     """
#     Create a new Queryer that will initialize embeddings and LLM
#     """

#     def __init__(self) -> None:
#         print(end="Initializing... ", flush=True)
#         self.embeddings = HuggingFaceEmbeddings()
#         repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
#         self.llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7)
#         self.chain = None
#         self.chat_history = []
#         print("DONE")

#     def process_file(self, file_path: str) -> None:
#         """
#         Pass the path to a PDF file for the Queryer to process.
#         It may take some time for large PDF's.
#         """
#         raw_text = ""
#         pdfreader = PdfReader(file_path)

#         for page in tqdm(pdfreader.pages):
#             content = page.extract_text()
#             if content:
#                 raw_text += content

#         text_splitter = RecursiveCharacterTextSplitter(
#             separators=["\n", "."],
#             chunk_size=800,
#             chunk_overlap=200,
#             length_function=len,
#         )
#         texts = text_splitter.split_text(raw_text)

#         print(end="Creating vector store... ", flush=True)
#         vector_store = FAISS.from_texts(texts, self.embeddings)
#         print("DONE!")

#         retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#         self.chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm, chain_type="stuff", retriever=retriever
#         )

#     def ask(self, query: str) -> str:
#         """
#         Pass a question for the Queryer. Returns the LLM's response.
#         """
#         if not query or query.isspace():
#             return None

#         # query += "\n\nPlease refrain from repeating yourself."
#         start_time = time.time()
#         print(end="Generating response...", flush=True)
#         response = self.chain.invoke({"question": query, "chat_history": self.chat_history})
#         response = response["answer"].strip()
#         self.chat_history.append((query, response))
#         print(f"DONE ({time.time() - start_time}s)")
#         return response
