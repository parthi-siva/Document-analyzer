import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import ArxivLoader

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from operator import itemgetter

from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)


try:
    from app.services.prompt_manager import get_model_config
except ImportError:
    # Fallback if prompt_manager is not available
    def get_model_config():
        return {"model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507", "temperature": 0}

import pandas as pd
from datasets import Dataset

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from typing import Dict, List, Any

# Define different retriever configurations to test
RETRIEVER_CONFIGS = {
    "basic": {"search_type": "similarity", "k": 2},
    "top_5": {"search_type": "similarity", "k": 5},
    "top_10": {"search_type": "similarity", "k": 10},
    "mmr_2": {"search_type": "mmr", "k": 2},
    "mmr_5": {"search_type": "mmr", "k": 5},
    "similarity_threshold": {
        "search_type": "similarity_score_threshold", 
        "score_threshold": 0.5,
        "k": 10
    },
}

SPLITTER_CONFIGS = {
    "small_chunks": {"chunk_size": 250, "chunk_overlap": 50},
    "medium_chunks": {"chunk_size": 500, "chunk_overlap": 100},
    "large_chunks": {"chunk_size": 1000, "chunk_overlap": 200},
}

# Load documents from local storage or fetch from ArXiv if not cached
def load_or_fetch_arxiv_docs():
    """Load documents from local cache or fetch from ArXiv if not available."""
    cache_file = "./arxiv_docs_cache.json"
    
    if os.path.exists(cache_file):
        print("Loading documents from local cache...")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        # Reconstruct Document objects
        from langchain.schema import Document
        docs = []
        for doc_data in cached_data:
            doc = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            docs.append(doc)
        return docs
    else:
        print("Fetching documents from ArXiv...")
        loader = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=5)
        docs = loader.load()
        
        # Cache the documents
        cache_data = []
        for doc in docs:
            cache_data.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        print(f"Cached {len(docs)} documents locally.")
        
        return docs

base_docs = load_or_fetch_arxiv_docs()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250)

docs = text_splitter.split_documents(base_docs)

vectorstore = Chroma(
            collection_name="test_collection",
            embedding_function=embedding_model,
            persist_directory="./test_db",
)
if vectorstore._collection.name is None:
    vectorstore.add_documents(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250)

base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 2})

template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

api_key = os.environ.get("OPENAI_API_KEY")
model_config = get_model_config()

primary_qa_llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepinfra.com/v1/openai",
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    temperature=0,
)

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

question_generation_llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepinfra.com/v1/openai",
    model=model_config.get("model_name", "Qwen/Qwen2.5-72B-Instruct"),
    temperature=0,
)

bare_prompt_template = "{content}"

bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)

question_schema = ResponseSchema(
    name="question",
    description="The question to ask the document",
)

question_response_schema = [question_schema]

question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schema)
format_instructions = question_output_parser.get_format_instructions()

qa_template = """\
You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

question_generation_chain = bare_template | question_generation_llm

# Generate or load cached Q&A triples
def generate_or_load_qac_triples():
    """Generate questions and answers or load from cache."""
    cache_file = "./qac_triples_cache.json"
    
    if os.path.exists(cache_file):
        print("Loading Q&A triples from cache...")
        with open(cache_file, 'r') as f:
            cached_qac = json.load(f)
        
        # Reconstruct Document objects for context and reference
        from langchain.schema import Document
        for triple in cached_qac:
            triple["context"] = Document(
                page_content=triple["context_content"],
                metadata=triple["context_metadata"]
            )
            triple["reference"] = Document(
                page_content=triple["reference_content"],
                metadata=triple["reference_metadata"]
            )
        return cached_qac
    else:
        print("Generating Q&A triples...")
        qac_triples = []
        
        for text in docs[:10]:
            messages = prompt_template.format_messages(
                context=text,
                format_instructions=format_instructions
            )
            response = question_generation_chain.invoke({"content" : messages})
            try:
                output_dict = question_output_parser.parse(response.content)
            except Exception as e:
                continue
            output_dict["context"] = text
            output_dict["reference"] = text
            qac_triples.append(output_dict)
        
        # Generate answers
        print("Generating answers...")
        # Setup answer generation components
        answer_generation_llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            temperature=0,
        )
        
        answer_schema = ResponseSchema(
            name="answer",
            description="an answer to the question"
        )
        
        answer_output_parser = StructuredOutputParser.from_response_schemas([answer_schema])
        answer_format_instructions = answer_output_parser.get_format_instructions()
        
        answer_template = """\
You are a University Professor creating a test for advanced students. For each question and context, create an answer.

answer: a answer about the context.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""
        
        answer_prompt_template = ChatPromptTemplate.from_template(template=answer_template)
        answer_generation_chain = bare_template | answer_generation_llm
        
        for triple in qac_triples:
            messages = answer_prompt_template.format_messages(
                context=triple["context"],
                question=triple["question"],
                format_instructions=answer_format_instructions
            )
            response = answer_generation_chain.invoke({"content" : messages})
            try:
                output_dict = answer_output_parser.parse(response.content)
            except Exception as e:
                continue
            triple["answer"] = output_dict["answer"]
        
        # Cache the triples
        cache_data = []
        for triple in qac_triples:
            cache_triple = {
                "question": triple["question"],
                "answer": triple.get("answer", ""),
                "context_content": triple["context"].page_content,
                "context_metadata": triple["context"].metadata,
                "reference_content": triple["reference"].page_content,
                "reference_metadata": triple["reference"].metadata
            }
            cache_data.append(cache_triple)
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        print(f"Cached {len(qac_triples)} Q&A triples.")
        
        return qac_triples

qac_triples = generate_or_load_qac_triples()

# Answer generation is now handled in the generate_or_load_qac_triples function
answer_generation_llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepinfra.com/v1/openai",
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    temperature=0,
)

answer_schema = ResponseSchema(
    name="answer",
    description="an answer to the question"
)

answer_response_schemas = [
    answer_schema,
]

answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)

ground_truth_qac_set = pd.DataFrame(qac_triples)
ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
ground_truth_qac_set["reference"] = ground_truth_qac_set["reference"].map(lambda x: str(x.page_content))
ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})

eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

def create_ragas_dataset(rag_pipeline, eval_dataset):
    rag_dataset = []
    for row in eval_dataset:
        answer = rag_pipeline.invoke({"question" : row["question"]})
        rag_dataset.append(
            {"question" : row["question"],
                "answer" : answer["response"].content,
                "contexts" : [context.page_content for context in answer["context"]],
                "ground_truths" : [row["ground_truth"]],
                "reference" : row["reference"]  # Add reference column required by context_precision
                }
        )
    rag_df = pd.DataFrame(rag_dataset)
    rag_eval_dataset = Dataset.from_pandas(rag_df)
    return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
    # Configure RAGAS to use DeepInfra LLM instead of default OpenAI
    deepinfra_llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
        model="Qwen/Qwen2.5-72B-Instruct",  # Use a smaller model for evaluation to save costs
        temperature=0,
    )
    
    # Wrap the LLM for RAGAS
    ragas_llm = LangchainLLMWrapper(deepinfra_llm)
    
    # Configure each metric to use our DeepInfra LLM
    context_precision.llm = ragas_llm
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    context_recall.llm = ragas_llm
    
    # Also configure embeddings to use the same model we're using elsewhere
    context_precision.embeddings = embedding_model
    faithfulness.embeddings = embedding_model
    answer_relevancy.embeddings = embedding_model
    context_recall.embeddings = embedding_model
    
    result = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    return result

def create_retriever_with_config(docs: List, retriever_config: Dict[str, Any]) -> Any:
    """Create a retriever with specific configuration."""
    # Create vector store from documents

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    if retriever_config["search_type"] == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_config["k"]}
        )
    elif retriever_config["search_type"] == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": retriever_config["k"], "fetch_k": retriever_config["k"] * 2}
        )
    elif retriever_config["search_type"] == "similarity_score_threshold":
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": retriever_config["score_threshold"],
                "k": retriever_config["k"]
            }
        )
    else:
        raise ValueError(f"Unknown search type: {retriever_config['search_type']}")

def create_rag_chain_with_retriever(retriever: Any) -> Any:
    """Create a RAG chain with a specific retriever."""
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
    )

def evaluate_retriever_configurations(
    base_docs: List, 
    eval_dataset: Dataset,
    retriever_configs: Dict[str, Dict[str, Any]],
    splitter_configs: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate different retriever configurations using RAGAS."""
    results = {}
    
    #if splitter_configs is None:
    splitter_configs = {"default": {"chunk_size": 250, "chunk_overlap": 50}}
    
    print(f"\n🔍 Starting Retriever Configuration Evaluation")
    print(f"Found {len(base_docs)} base documents")
    print(f"Testing {len(retriever_configs)} retriever configs with {len(splitter_configs)} splitter configs")
    
    for splitter_name, splitter_config in splitter_configs.items():
        print(f"\nProcessing splitter: {splitter_name}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitter_config["chunk_size"],
            chunk_overlap=splitter_config["chunk_overlap"]
        )
        
        current_docs = text_splitter.split_documents(base_docs)
        print(f"  Created {len(current_docs)} chunks")
        
        for retriever_name, retriever_config in retriever_configs.items():
            config_key = f"{splitter_name}_{retriever_name}"
            print(f"  Testing: {config_key}")
            
            try:
                retriever = create_retriever_with_config(current_docs, retriever_config)
                
                rag_chain = create_rag_chain_with_retriever(retriever)
                
                ragas_dataset = create_ragas_dataset(rag_chain, eval_dataset)
                
                evaluation_result = evaluate_ragas_dataset(ragas_dataset)
                
                results[config_key] = {
                    "splitter_config": splitter_config,
                    "retriever_config": retriever_config,
                    "evaluation_metrics": evaluation_result,
                    "num_chunks": len(current_docs),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"  Failed: {str(e)}")
                results[config_key] = {
                    "splitter_config": splitter_config,
                    "retriever_config": retriever_config,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

    return results

def save_evaluation_results(results: Dict[str, Any], filename: str = "retriever_evaluation_results.json"):
    """Save evaluation results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n Results saved to {filename}")

def create_results_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary DataFrame from evaluation results."""
    summary_data = []
    
    for config_name, result in results.items():
        if "error" in result:
            continue

        metrics = result["evaluation_metrics"]
        retriever_config = result["retriever_config"]
        splitter_config = result["splitter_config"]

        summary_data.append({
            "config_name": config_name,
            "search_type": retriever_config.get("search_type", "unknown"),
            "k": retriever_config.get("k", "N/A"),
            "chunk_size": splitter_config["chunk_size"],
            "chunk_overlap": splitter_config["chunk_overlap"],
            "num_chunks": result["num_chunks"],
            "context_precision": metrics["context_precision"],
            "faithfulness": metrics["faithfulness"],
            "answer_relevancy": metrics["answer_relevancy"],
            "context_recall": metrics["context_recall"]
        })

    df = pd.DataFrame(summary_data)

    if not df.empty:
        df["context_precision"] = pd.to_numeric(df["context_precision"], errors='coerce')
        df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors='coerce')
        df["answer_relevancy"] = pd.to_numeric(df["answer_relevancy"], errors='coerce')
        df["context_recall"] = pd.to_numeric(df["context_recall"], errors='coerce')

        df["composite_score"] = (
            df["context_precision"] * 0.25 +
            df["faithfulness"] * 0.25 +
            df["answer_relevancy"] * 0.25 +
            df["context_recall"] * 0.25
        )
        df = df.sort_values("composite_score", ascending=False)

    return df
