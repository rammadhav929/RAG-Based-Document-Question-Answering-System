# RAG-Based Document Question Answering System | Python, FastAPI, FAISS, SentenceTransformers, HuggingFace
- Architected a Retrieval Augmented Generation (RAG) pipeline for semantic QA and summarization over PDF document
- Engineered document ingestion workflow with text extraction, semantic chunking, and embedding generation using SentenceTransformers
- Constructed FAISS-based vector retrieval layer to surface contextually relevant document segments
- Leveraged Meta Llama-3.1-8B-Instruct via HuggingFace for grounded context-aware response generation
- Orchestrated FastAPI service layer for document upload, indexing lifecycle, and real-time query execution
- Optimized retrieval latency through persistent vector indexing and elimination of redundant embedding computation

# Libraries:-

fitz (PyMuPDF): Used to open, read, and extract text or images from PDF documents.

faiss: A library developed by Meta for efficient similarity search and clustering of dense vectors.

numpy: The fundamental package for scientific computing, handles and format data as numerical arrays.

os: A standard Python module used to interact with the operating system.

sentence_transformers: Provides easy access to pre-trained models that turn sentences or paragraphs into meaningful vector embeddings.

# all-MiniLM-L6-v2 model :-
- This convert text into vector format.
- These numbers represent the semantic meaning of the text, allowing a computer to calculate how similar two different sentences are mathematically.
- L6 represent number of internal layers.


# SEMANTIC CHUNK 
Exactly—the overlap acts as a semantic bridge that ensures the relationship between ideas isn't severed by a hard cut. By repeating those 40 words, you provide the model with the "before and after" context needed to accurately understand the full meaning of a passage.

# Build index
 This function automates the creation of a searchable "brain" for a PDF by extracting its text, slicing it into chunks, and converting those chunks into numerical vectors. These vectors are then stored in a FAISS index, which is a specialized database that allows for ultra-fast similarity searches later.

