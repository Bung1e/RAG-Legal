import fitz
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from tqdm import tqdm
import torch

PDF_PATH = "preprocessed/grazd.pdf"
DB_PATH = "chroma_db_structured"
COLLECTION_NAME = "belarus_civil_code"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
SEPARATORS = ["\n\n", "\n"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def extract_articles_from_pdf(pdf_path):
    """
    Извлекает текст из PDF и разделяет его на статьи.

    Args:
        pdf_path (str): Путь к PDF файлу.

    Returns:
        list[dict]: Список словарей, где каждый словарь содержит
                    'article_number' (str) и 'text' (str) для одной статьи.
                    Возвращает пустой список, если статьи не найдены.
    """
    articles = []
    full_text = ""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        for page in tqdm(doc, desc="Reading PDF pages"):
            full_text += page.get_text("text", sort=True) + "\n"
    except Exception as e:
        print(f"Error opening or reading PDF: {e}")
        return []
    finally:
        if doc:
            doc.close()

    article_pattern = re.compile(
        r"^\s*(Статья(?:\s|\u00a0)+[\d\.]+)\.?\s*(.*?)\n(.*?)(?=^\s*Статья(?:\s|\u00a0)+[\d\.]+\.?\s*|\Z)",
        re.MULTILINE | re.DOTALL
    )

    matches = article_pattern.finditer(full_text)
    last_end = 0

    extracted_articles = []
    for match in tqdm(matches, desc="Extracting articles"):
        article_header = match.group(1).strip() # "Статья XXX"
        article_title = match.group(2).strip()  # Название статьи
        article_body = match.group(3).strip()   # Текст статьи

        # Извлекаем номер статьи
        num_match = re.search(r'(\d+(\.\d+)*)', article_header)
        article_number = num_match.group(1) if num_match else "Unknown"

        article_full_text = f"{article_header}. {article_title}\n{article_body}"

        extracted_articles.append({
            "article_number": article_number,
            "text": article_full_text,
            "title": article_title
        })
        last_end = match.end()

    if not extracted_articles and full_text:
         print("Warning: Could not extract articles using regex. Treating the whole document as one chunk (not recommended).")
         return [] # Возвращаем пусто, т.к. структура не найдена


    print(f"Successfully extracted {len(extracted_articles)} articles.")
    return extracted_articles

def create_annotated_chunks(articles, chunk_size, chunk_overlap, separators):
    """
    Разбивает текст каждой статьи на чанки и добавляет аннотации.

    Args:
        articles (list[dict]): Список словарей статей.
        chunk_size (int): Максимальный размер чанка.
        chunk_overlap (int): Перекрытие между чанками.
        separators (list[str]): Разделители для сплиттера.
    Returns:
        tuple: Кортеж из трех списков:
               - chunk_texts (list[str]): Аннотированные тексты чанков.
               - chunk_metadatas (list[dict]): Метаданные для каждого чанка.
               - chunk_ids (list[str]): Уникальные ID для каждого чанка.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=separators,
    )

    all_chunk_texts = []
    all_metadatas = []
    all_ids = []

    for article in tqdm(articles, desc="Chunking articles"):
        article_num = article["article_number"]
        article_text = article["text"]
        article_title = article.get("title", "") # Получаем название, если есть

        chunks = text_splitter.split_text(article_text)

        for i, chunk in enumerate(chunks):
            annotation = f"Статья {article_num}. {article_title}\n\n"
            annotated_text = annotation + chunk

            metadata = {
                "source_pdf": PDF_PATH,
                "article": str(article_num),
                 "article_title": article_title,
                "chunk_index": i 
            }

            chunk_id = f"article_{article_num}_chunk_{i}_{uuid.uuid4()}"

            all_chunk_texts.append(annotated_text)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

    print(f"Created {len(all_chunk_texts)} annotated chunks.")
    return all_chunk_texts, all_metadatas, all_ids

if __name__ == "__main__":
    print("Starting PDF processing...")

    # 1. Извлечение статей из PDF
    articles = extract_articles_from_pdf(PDF_PATH)

    if not articles:
        print("No articles extracted. Exiting.")
        exit()

    # 2. Создание аннотированных чанков
    chunk_texts, chunk_metadatas, chunk_ids = create_annotated_chunks(
        articles, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
    )

    if not chunk_texts:
        print("No chunks created. Exiting.")
        exit()

    # 3. Инициализация эмбеддера
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME} on {DEVICE}")
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device=DEVICE
    )
    print("Embedding model initialized.")


    # 4. Инициализация и заполнение ChromaDB
    print(f"Initializing ChromaDB persistent client at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    print(f"Getting or creating collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=hf_ef, 
        metadata={"hnsw:space": "cosine"}
    )
    print("Collection ready.")
    batch_size = 100 #
    total_batches = (len(chunk_ids) + batch_size - 1) // batch_size

    print(f"Adding {len(chunk_ids)} chunks to ChromaDB in {total_batches} batches...")
    for i in tqdm(range(0, len(chunk_ids), batch_size), desc="Adding batches to Chroma"):
        batch_ids = chunk_ids[i:i + batch_size]
        batch_texts = chunk_texts[i:i + batch_size]
        batch_metadatas = chunk_metadatas[i:i + batch_size]

        try:
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
        except Exception as e:
            print(f"\nError adding batch starting at index {i}: {e}")

    print("-" * 30)
    print("Processing finished successfully!")
    print(f"Processed PDF: {PDF_PATH}")
    print(f"Total articles extracted: {len(articles)}")
    print(f"Total chunks created and added to ChromaDB: {collection.count()}")
    print(f"Database stored at: {DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    print("-" * 30)
