#data_loader.py

import os
from langchain_community.document_loaders import PyPDFLoader #, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


#pdf loader
def process_all_pdfs(pdf_directory):
    """process all pdf files from directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    #find all pdf files recursively and list them
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    
    for pdf_file in pdf_files:
        print(f"\nProcessing:{pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            #add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"loaded: {len(documents)} pages")
        
        except Exception as e:
            print(f" error:{e}")

    print(f"\nTotal documents loaded:{len(all_documents)}")
    return all_documents

# all_pdf_documents = process_all_pdfs("data/")

# if all_pdf_documents:
#     print("pdfs processed")


#chunking the pdf 
def split_documents(documents):
    """Split documents into smaller chunks for better vector embeddings clustering"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200, 
        chunk_overlap = 50,  #contains the few words that from the previous chunk, that makes it relative to each other
        length_function = len,
        separators = ["\n\n","\n"," ",""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"split {len(documents)} documents into {len(split_docs)} chunk")

    if split_docs:
        print(f"\nExample chunk:")
        print(f"content: {split_docs[0].page_content[:100]}...")
        print(f"metadata: {split_docs[0].metadata}")

    return split_docs

# chunks = split_documents(all_pdf_documents)
# if chunks:
#     print("chunks done")








