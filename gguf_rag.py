import os

HF_CACHE_PATH = "E:\\huggingface_cache" # Укажи свой путь
os.makedirs(HF_CACHE_PATH, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_PATH
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_PATH
os.environ['SENTENCE_TRANSFORMERS_HOME'] = HF_CACHE_PATH
os.environ['LLAMA_CPP_CACHE_PATH'] = os.path.join(HF_CACHE_PATH, 'llama_cpp')
os.makedirs(os.environ['LLAMA_CPP_CACHE_PATH'], exist_ok=True)

import torch
import chromadb
from sentence_transformers import SentenceTransformer # Оставляем для эмбеддинга запроса
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_cpp import Llama
import time
import warnings

# --- Установка директории кэша (для эмбеддера) ---


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Конфигурация ---
DB_PATH = "chroma_db_structured"
COLLECTION_NAME = "belarus_civil_code"

# Модель для эмбеддинга
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

GGUF_REPO_ID = "google/gemma-3-4b-it-qat-q4_0-gguf" # Репозиторий на Hugging Face
GGUF_FILENAME_PATTERN = "gemma-3-4b-it-q4_0.gguf"      # Паттерн для файла нужной квантизации

N_GPU_LAYERS = 20  # Количество слоев для выгрузки на GPU.
                   # -1 = выгрузить все возможное.
                   # 0 = использовать только CPU.
N_CTX = 8192      

SEARCH_K = 5
MAX_NEW_TOKENS = 512
GENERATION_TEMPERATURE = 0.1


def initialize_simple_rag():
    """Загружает эмбеддер, GGUF LLM (через from_pretrained) и подключается к БД."""
    embedder_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedder will use device: {embedder_device}")

    try:
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=embedder_device)
        print("Embedding Model loaded.")

        print(f"Loading GGUF Model from repo: {GGUF_REPO_ID}, filename: {GGUF_FILENAME_PATTERN}")
        llm_cpp = Llama.from_pretrained(
            repo_id=GGUF_REPO_ID,
            filename=GGUF_FILENAME_PATTERN,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=512,
            verbose=True # Полезно для отладки загрузки и offloading
        )
        print("GGUF LLM loaded.")

        # Подключение к БД (без изменений)
        print(f"Connecting to DB: {DB_PATH}/{COLLECTION_NAME}")
        db_client = chromadb.PersistentClient(path=DB_PATH)
        collection = db_client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to DB with {collection.count()} documents.")
        print("-" * 38 + "\n")

        # --- ИЗМЕНЕНИЕ: Возвращаем embedder, llm_cpp, collection, embedder_device ---
        return embedder, llm_cpp, collection, embedder_device

    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}")
        raise RuntimeError(f"Failed to initialize components: {e}")

# --- Функции Простого RAG ---
def simple_retrieve(query, embedder, collection, n_results, embedder_device):
    """Простой поиск в ChromaDB без ранжирования."""
    try:
        # Используем правильное устройство для эмбеддера
        query_embedding = embedder.encode(query, convert_to_tensor=True, device=embedder_device)
        results = collection.query(
            query_embeddings=[query_embedding.cpu().numpy()],
            n_results=n_results,
            include=['documents', 'metadatas']
        )
        retrieved_chunks = []
        if results and results.get('ids') and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                retrieved_chunks.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
        # Возвращаем только чанки, эмбеддинг не нужен дальше
        return retrieved_chunks
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def format_simple_context(chunks):
    context = ""
    sources = set()
    return context.strip(), sorted(list(sources))

def build_simple_prompt(query, context):
    system_prompt = """Ты — ИИ-ассистент по Гражданскому Кодексу Республики Беларусь"""
    return prompt

def generate_simple_answer(prompt, llm_cpp_model, max_new_tokens, temperature):
    try:
        output = llm_cpp_model(
            prompt,
            max_tokens=max_new_tokens, 
            temperature=temperature,
            stop=["\nВОПРОС:", "\nЗАПРОС ПОЛЬЗОВАТЕЛЯ:"],
            echo=False # Не печатать промпт в ответе
        )
        answer = output['choices'][0]['text'].strip()
        return answer
    except Exception as e:
        print(f"Error during LLM generation (llama-cpp): {e}")
        return f"Ошибка генерации ответа Llama CPP: {e}"

if __name__ == "__main__":
    try:
        s_embedder, s_llm_cpp, s_collection, s_embedder_device = initialize_simple_rag()
        s_is_ready = True
    except RuntimeError as e:
        print(e); s_is_ready = False

    if s_is_ready:
        print("\n--- Simple RAG CLI Ready (GGUF via Llama.from_pretrained) ---")
        print("Используется модель из репозитория:", GGUF_REPO_ID)
        print(f"Извлекается {SEARCH_K} чанков без ранжирования.")
        print(f"Llama CPP n_gpu_layers: {N_GPU_LAYERS}")
        print("Введите ваш вопрос или 'exit' для выхода.")

        while True:
            s_query = input("\nВаш вопрос: ")
            if s_query.lower() == 'exit': break
            if not s_query: continue

            start_time = time.time()
            try:
                # 1. Retrieve (передаем s_embedder_device)
                s_retrieved_chunks = simple_retrieve(s_query, s_embedder, s_collection, SEARCH_K, s_embedder_device)
                if not s_retrieved_chunks:
                    print("Не удалось найти релевантные документы в базе.")
                    continue

                # 2. Format Context
                s_context_str, s_sources = format_simple_context(s_retrieved_chunks)

                # 3. Build Prompt
                prompt = build_simple_prompt(s_query, s_context_str)

                # 4. Generate Answer
                print("\nГенерация ответа (Llama CPP)...")
                s_answer = generate_simple_answer(prompt, s_llm_cpp, MAX_NEW_TOKENS, GENERATION_TEMPERATURE)
                end_time = time.time()

                # 5. Display Result 
                print("\n--- Ответ ---")
                print(s_answer)
                print(f"\nВремя обработки запроса: {end_time - start_time:.2f} сек.")

            except Exception as e:
                print(f"Ошибка при обработке запроса: {e}")
            finally:
                 print("-" * 50)
    else:
        print("Simple RAG CLI could not start due to initialization errors.")

    print("Завершение работы.")
