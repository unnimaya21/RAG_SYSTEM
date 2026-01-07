from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data"
PERSIST_DIRECTORY = "chroma_db"


def ingest_documents():
    print("📥 Loading documents...")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    documents = loader.load()

    if not documents:
        raise ValueError("❌ No documents found")

    print(f"📄 Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("❌ No chunks created")

    print(f"✂️ Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print("✅ Documents ingested into Chroma")
