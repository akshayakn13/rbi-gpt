import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_collection(db_path="rbi_db"):
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="rbi_master_circulars")
    return collection

def search_rbi(collection, question, k=20):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def ask_rbi(collection, gemini_client, question, simple=False, chat_history=None):
    results = search_rbi(collection, question)
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[Source: {meta['title']}]\n{doc}")
    context = "\n\n".join(context_parts)

    if simple:
        tone = "Explain in simple, plain language that a non-banker can understand. Avoid jargon."
    else:
        tone = "Use precise regulatory language suitable for banking professionals."

    # Build conversation history string
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:  # last 6 messages = 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

    prompt = f"""You are an RBI regulatory assistant.
Answer ONLY using the provided RBI context below.
{tone}
Always mention which circular or direction the answer comes from.
If the answer is not present in the context, say:
'The information is not available in the retrieved RBI documents.'

{f'Previous conversation:{chr(10)}{history_text}' if history_text else ''}

Context:
{context}

Current Question:
{question}
"""
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text