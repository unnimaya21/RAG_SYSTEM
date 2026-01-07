from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(path: str):
    loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    return loader.load()
