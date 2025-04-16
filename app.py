import gradio as gr
import time
import os

# Если модель не скачана то раскоментировать и указать путь к кэшу
# HF_CACHE_PATH = "ваш путь к кешу"
# os.makedirs(HF_CACHE_PATH, exist_ok=True)
# os.environ['HF_HOME'] = HF_CACHE_PATH

import torch
import docx
import uuid
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gguf_rag import (
    initialize_simple_rag,
    simple_retrieve as retrieve_code_chunks,
    generate_simple_answer,
    SEARCH_K,
    MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE,
    N_GPU_LAYERS
)

try:
    embedder, llm_cpp, belarus_code_collection, embedder_device = initialize_simple_rag()
    models_loaded = True
    print(f"--- RAG Backend Initialized Successfully (Embedder: {embedder_device}, Llama CPP Layers on GPU: {N_GPU_LAYERS}) ---")
except RuntimeError as e:
    print(f"FATAL ERROR during backend initialization: {e}")
    embedder, llm_cpp, belarus_code_collection, embedder_device = None, None, None, None
    models_loaded = False

DOC_CHUNK_SIZE = 1000
DOC_CHUNK_OVERLAP = 200
DOC_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! "]
SEARCH_K_DOC = 5  

def read_docx(file_path):
    try: doc = docx.Document(file_path); return '\n'.join([p.text for p in doc.paragraphs])
    except Exception as e: print(f"Err DOCX: {e}"); return None

def chunk_text(text, chunk_size, chunk_overlap, separators):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, is_separator_regex=False, separators=separators
    )
    return splitter.split_text(text)

def process_uploaded_docx(file_obj, progress=gr.Progress()):
    if not models_loaded or embedder is None: return None, "Ошибка: Эмбеддер не загружен."
    if not file_obj: return None, "Файл не загружен."

    file_path = file_obj.name
    progress(0, desc="Чтение...")
    doc_text = read_docx(file_path)
    if not doc_text: return None, "Ошибка чтения или пустой файл."

    progress(0.2, desc="Чанкинг...")
    doc_chunks_text = chunk_text(doc_text, DOC_CHUNK_SIZE, DOC_CHUNK_OVERLAP, DOC_SEPARATORS)
    if not doc_chunks_text: return None, "Ошибка разбиения на чанки."

    progress(0.4, desc="Эмбеддинг...")
    try:
        chunk_embeddings = embedder.encode(
            doc_chunks_text, convert_to_tensor=True, show_progress_bar=True, device=embedder_device
        ).cpu().numpy()
    except Exception as e: print(f"Error embedding: {e}"); return None, f"Ошибка эмбеддинга: {e}"

    processed_data = [{
        "id": f"doc_chunk_{uuid.uuid4()}", "text": text, "embedding": chunk_embeddings[i],
        "metadata": {"source": "uploaded_document", "chunk_index": i}
    } for i, text in enumerate(doc_chunks_text)]

    progress(1, desc="Готово!")
    return processed_data, f"Документ '{os.path.basename(file_path)}' обработан ({len(processed_data)} чанков)."

def retrieve_from_list(query_embedding, processed_doc_data, n_results):
    if not processed_doc_data or embedder_device is None: return []
    try:
        doc_embeddings = torch.tensor([data['embedding'] for data in processed_doc_data]).to(embedder_device)
        query_embedding_tensor = torch.tensor(query_embedding).to(embedder_device)
        similarities = torch.nn.functional.cosine_similarity(query_embedding_tensor, doc_embeddings)
        top_k_indices = torch.topk(similarities, k=min(n_results, len(doc_embeddings))).indices.cpu().numpy()
        return [processed_doc_data[i] for i in top_k_indices]
    except Exception as e: print(f"Error retrieving from list: {e}"); return []

def format_combined_context(code_chunks, doc_chunks):
    """Форматирует контекст из Законодательства и Документа."""
    context = ""
    sources_code = set(); sources_doc = set()
    # Сначала документ
    for i, chunk in enumerate(doc_chunks):
        metadata = chunk.get('metadata', {})
        if metadata.get('source') == 'uploaded_document':
            chunk_idx = metadata.get('chunk_index', '?')
            source_str = f"Фрагмент документа {chunk_idx+1}"
            sources_doc.add(source_str)
            context += f"--- Контекст из Документа ({source_str}) ---\\n{chunk.get('text', '')}\\n\\n"
    # Затем кодексы
    for i, chunk in enumerate(code_chunks):
        metadata = chunk.get('metadata', {})
        code_name = metadata.get('source_code', 'Законодательство РБ')
        article_num = metadata.get('article', '?')
        article_title = metadata.get('article_title', '')
        source_str = f"{code_name}, Статья {article_num}{f'. {article_title}' if article_title else ''}"
        sources_code.add(source_str)
        context += f"--- Контекст из Законодательства ({source_str}) ---\\n{chunk.get('text', '')}\\n\\n"
    return context.strip(), sorted(list(sources_code)), sorted(list(sources_doc))

system_prompt_doc = """Ты — высококвалифицированный русскоязычный ИИ-ассистент юриста. Твоя задача - анализировать предоставленный пользователем документ (КОНТЕКСТ ИЗ ДОКУМЕНТА) в свете законодательства Республики Беларусь (КОНТЕКСТ ИЗ ЗАКОНОДАТЕЛЬСТВА).
- Внимательно изучи ЗАПРОС ПОЛЬЗОВАТЕЛЯ.
- Сопоставь пункты документа с релевантными статьями законодательства из контекста.
- Укажи на возможные несоответствия, риски или особенности документа, основываясь СТРОГО на предоставленном КОНТЕКСТЕ (из документа и законодательства).
- Если КОНТЕКСТ не позволяет ответить на вопрос или провести анализ, сообщи об этом честно. Не домысливай.
- Цитируй конкретные пункты документа и номера статей законодательства в своем ответе для обоснования.
- Ответ должен быть четким, структурированным и по существу запроса."""

system_prompt_code = """Ты — ИИ-ассистент по законодательству Республики Беларусь. Твоя задача - отвечать на вопросы пользователя на основе предоставленного контекста из Кодекса.
- Отвечай только на основе приведенного ниже КОНТЕКСТА ИЗ ЗАКОНОДАТЕЛЬСТВА.
- Если КОНТЕКСТ не содержит ответа на ЗАПРОС ПОЛЬЗОВАТЕЛЯ, сообщи об этом честно. Не придумывай информацию.
- Цитируй номера статей из КОНТЕКСТА, если это уместно для ответа.
- Формулируй ответ четко, ясно и по существу запроса. Используй русский язык.""" 

def build_rag_prompt(query, context, system_prompt):
    prompt = f"""{system_prompt}

КОНТЕКСТ:
{context}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ОТВЕТ АССИСТЕНТА:
"""
    return prompt

# Gradio
def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def bot_response(history, processed_doc_state):

    query = history[-1][0]
    print(f"\\nProcessing query: {query}")
    history[-1][1] = ""

    analyze_keywords = ["договор", "контракт", "оцен", "соответств", "пункт", "услови", "документ", "проанализир"]
    is_doc_analysis = processed_doc_state is not None and any(k in query.lower() for k in analyze_keywords)
    print(f"[Debug] Is document analysis request: {is_doc_analysis}")

    # Определение целевого запроса к Кодексам
    doc_type = None
    code_search_query = query
    if is_doc_analysis and processed_doc_state:
        first_chunk_text = processed_doc_state[0].get("text", "").lower()
        if "договор займа" in first_chunk_text: doc_type="займа"
        if doc_type: code_search_query = f"статьи законодательства рб договор {doc_type}"
        else: code_search_query = "общие положения о договорах законодательство рб"

    start_time = time.time()

    try:
        # Шаг 1: Retrieve
        code_chunks = retrieve_code_chunks(
            code_search_query, embedder, belarus_code_collection,
            SEARCH_K,
            embedder_device
        )
        print(f"Retrieved {len(code_chunks)} chunks from Code DB.")

        doc_chunks = []
        query_embedding = None
        if is_doc_analysis:
            # Вычисляем эмбеддинг ОРИГИНАЛЬНОГО запроса
            try:
                query_embedding = embedder.encode(query, convert_to_tensor=True, device=embedder_device).cpu().numpy()
            except Exception as emb_err: print(f"[Error] Failed to encode query: {emb_err}")

            if query_embedding is not None:
                 doc_chunks = retrieve_from_list(query_embedding, processed_doc_state, SEARCH_K_DOC)
                 print(f"Retrieved {len(doc_chunks)} chunks from Document.")

        # Шаг 2: Format Context
        context_str, sources_code_list, sources_doc_list = format_combined_context(code_chunks, doc_chunks)
        # --- ДОБАВЛЯЕМ ФОРМИРОВАНИЕ sources_str ---
        sources_str = ""
        if sources_code_list:
            sources_str += "\n*Источники из Законодательства:*\n"
            for src in sources_code_list:
                sources_str += f"- {src}\n"
        if sources_doc_list:
            sources_str += "\n*Источники из Документа:*\n"
            for src in sources_doc_list:
                sources_str += f"- {src}\n"
        sources_str = sources_str.strip()

        # Шаг 3: Build Prompt
        system_prompt = system_prompt_doc if is_doc_analysis else system_prompt_code
        prompt = build_rag_prompt(query, context_str, system_prompt)

        # Шаг 4: Generate Answer
        print("\\nGenerating response (Llama CPP)...")
        answer = generate_simple_answer(prompt, llm_cpp, MAX_NEW_TOKENS, GENERATION_TEMPERATURE)
        end_time = time.time()
        response_time_str = f"\\n\\n*(Время ответа: {end_time - start_time:.2f} сек.)*"
        final_answer = f"{answer}\\n\\n---\\n{sources_str}{response_time_str}"

    except Exception as e:
        end_time = time.time()
        final_answer = f"Произошла ошибка: {e}\\n\\n*(Время до ошибки: {end_time - start_time:.2f} сек.)*"

    history[-1][1] = final_answer
    return history

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    processed_doc_state = gr.State(None)
    gr.Markdown("# Юридический ИИ-Ассистент (Законодательство РБ + Анализ Документов) - GGUF")
    gr.Markdown("Задайте вопрос по законодательству РБ или загрузите DOCX-файл для анализа.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False, height=600)
            txt_msg = gr.Textbox(label="Ваш вопрос:", placeholder="Введите вопрос...", lines=2)
            submit_btn = gr.Button("Отправить", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### Загрузка документа (DOCX)")
            file_input = gr.File(label="Загрузить документ (.docx)", file_types=['.docx'], type="filepath")
            upload_status = gr.Textbox(label="Статус обработки", interactive=False)
            clear_doc_btn = gr.Button("Очистить документ")

    # --- Обработчики Событий ---
    file_input.upload(process_uploaded_docx, inputs=[file_input], outputs=[processed_doc_state, upload_status])
    def clear_document(): return None, "Загруженный документ очищен."
    clear_doc_btn.click(clear_document, outputs=[processed_doc_state, upload_status])

    txt_msg.submit(add_text, [chatbot, txt_msg], [chatbot, txt_msg], queue=False).then(
        bot_response, [chatbot, processed_doc_state], chatbot
    ).then(lambda: gr.update(interactive=True), None, [txt_msg], queue=False)

    submit_btn.click(add_text, [chatbot, txt_msg], [chatbot, txt_msg], queue=False).then(
        bot_response, [chatbot, processed_doc_state], chatbot
    ).then(lambda: gr.update(interactive=True), None, [txt_msg], queue=False)

if __name__ == "__main__":
    if models_loaded:
        print("\\nLaunching Gradio Interface (using gguf_rag backend)...")
        demo.queue().launch(share=False)
    else:
        print("\\nGradio Interface cannot launch due to backend initialization errors.")
