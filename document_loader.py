import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXT = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "markdown"
}

def load_documents(directory: str) -> List[Document]:
    all_docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXT:
                continue
            file_path = os.path.join(root, file)
            
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".md":
                # Markdown 本质为文本，使用 TextLoader 更稳定，避免 unstructured/spaCy 依赖问题
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:  # .txt
                loader = TextLoader(file_path, autodetect_encoding=True)
            
            # 加载文档
            try:
                docs = loader.load()
            except Exception as e:
                logger.warning(f"加载文件失败: {file_path}, 错误: {e}")
                continue

            # 为每个文档片段追加/确保元数据
            for doc in docs:
                # 保留 loader 自带的元数据（如 PyPDFLoader 的 source、page）
                # 仅在缺失时才补充 source
                doc.metadata.setdefault("source", file_path)
                doc.metadata["file_type"] = SUPPORTED_EXT[ext]
            all_docs.extend(docs)
    
    return all_docs