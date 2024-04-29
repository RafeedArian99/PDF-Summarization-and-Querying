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

        self.map_template = map_template
        self.reduce_template = reduce_template
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

        self.llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})



        map_prompt = PromptTemplate.from_template(self.map_template)
        self.map_chain = LLMChain(llm=self.llm, prompt=map_prompt)


        
        reduce_prompt = PromptTemplate.from_template(self.reduce_template)

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