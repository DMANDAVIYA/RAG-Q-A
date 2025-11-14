import os
import sys
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


def load_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def split_text(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=". "
    )
    return text_splitter.split_documents(documents)


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks, embeddings, persist_dir="./chroma_db"):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )


def initialize_llm(model_name="mistral"):
    return Ollama(model=model_name)


def create_qa_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )


def verify_ollama():
    try:
        test_llm = Ollama(model="mistral")
        test_llm.invoke("test")
        return True
    except Exception as e:
        print(f"\nError: Ollama is not running or Mistral model is not available.")
        print(f"Details: {str(e)}")
        print("\nPlease ensure:")
        print("1. Ollama is installed and running")
        print("2. Mistral model is pulled: ollama pull mistral")
        return False


def setup_rag_pipeline(file_path="speech.txt"):
    print("Loading document...")
    documents = load_document(file_path)
    
    print("Splitting text into chunks...")
    chunks = split_text(documents)
    
    print("Creating embeddings...")
    embeddings = create_embeddings()
    
    print("Building vector store...")
    vector_store = create_vector_store(chunks, embeddings)
    
    print("Initializing LLM...")
    llm = initialize_llm()
    
    print("Creating QA chain...")
    qa_chain = create_qa_chain(vector_store, llm)
    
    return qa_chain


def run_qa_system():
    print("\n" + "="*60)
    print("AmbedkarGPT - Q&A System")
    print("="*60)
    
    if not verify_ollama():
        sys.exit(1)
    
    try:
        qa_chain = setup_rag_pipeline()
        print("\nSetup complete! Ready to answer questions.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using AmbedkarGPT!")
                break
            
            if not question:
                continue
            
            print("\nThinking...")
            response = qa_chain.invoke({"query": question})
            print(f"\nAnswer: {response['result']}")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure 'speech.txt' is in the same directory as this script.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_qa_system()
