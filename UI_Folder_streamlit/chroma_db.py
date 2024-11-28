import chromadb # type: ignore

# Initialize ChromaDB client


# # Connect to the collection
# collection_name = "your_collection_name"  # Replace with your collection's name
# collection = client.get_collection(collection_name)

# # Example: Retrieve all documents from the collection
# all_docs = collection.get()


def collection():
    client = chromadb.PersistentClient(path="./chroma_storage")
    collection_name = "rag_embedding_collection"  # Your collection's name
    collection = client.get_collection(name=collection_name)
    return collection
