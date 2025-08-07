from langchain_core.documents import Document

def document_splitter(text_splitter, result):
    all_chunks = []
    for doc in result:
        # Get the text content
        text = doc.text

        # Split the text into chunks
        chunks = text_splitter.split_text(text)

        # Convert to LangChain Document objects
        for i, chunk in enumerate(chunks):
            langchain_doc = Document(
                page_content=chunk,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": i,
                    "original_doc_id": doc.id_ if hasattr(doc, 'id_') else None
                }
            )
            all_chunks.append(langchain_doc)
    return all_chunks


