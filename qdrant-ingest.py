from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import logging

def load_and_split_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(content)
        documents = [Document(page_content=text) for text in texts]
        return documents
    except Exception as e:
        logging.error(f"Error loading and splitting text file {file_path}: {e}")
        return []

def main():
    try:
        # Assuming text files are located in a directory called "text_files"
        directory = r"C:\Users\srika\OneDrive\Desktop\New folder\final_planty\data"
        all_documents = []
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                documents = load_and_split_text(file_path)
                all_documents.extend(documents)
        
        if not all_documents:
            logging.error("No text files found or all text files are empty.")
            return
        
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        url = "http://localhost:6333"
        qdrant = Qdrant.from_documents(
            all_documents,
            embeddings,
            url=url,
            prefer_grpc=False,
            collection_name="planty"
        )

        print("Collection successfully created!")
    except Exception as e:
        logging.error(f"Error during the process: {e}")

if __name__ == "__main__":
    main()
