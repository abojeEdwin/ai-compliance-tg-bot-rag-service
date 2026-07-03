import os
import cohere
from dotenv import load_dotenv

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

def get_bulk_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates 384-dimension vector float dimensions in bulk arrays using Cohere."""
    if not texts:
        return []
        
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    
    return response.embeddings.float

def get_query_embedding(text: str) -> list[float]:
    """Generates a single search vector optimized for matching user queries."""
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    return response.embeddings.float[0]

def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Passes context blocks to Command-R for a bounded, grounded answer."""
    joined_context = "\n---\n".join(context_chunks)
    
    prompt = (
        "You are an expert technical and business automation assistant for Youverify.\n"
        "Your task is to answer questions about integrating Youverify APIs, SDKs, and products.\n\n"
        
        "CRITICAL INSTRUCTION FOR DEVICE FINGERPRINTING/TELEMETRY CONTEXT:\n"
        "If the provided context contains technical browser telemetry terms (like WebGL, Canvas, extensions, "
        "or hardware components), DO NOT explain what those graphics concepts are generally.\n"
        "Instead, contextually explain that these variables are part of Youverify's 'Signal Management' and "
        "fraud detection payloads returned by endpoints like 'Get Signals using ID' to flag high-risk or automated bot devices during verification.\n\n"
        
        "General Rules:\n"
        "1. Answer using ONLY the verified context provided below.\n"
        "2. If the answer cannot be confidently derived from the context, say: 'I am sorry, I do not have "
        "that specific detail on hand. Let me loop in a team member to assist you.' Do not invent info.\n\n"
        f"VERIFIED CONTEXT:\n{joined_context}"
    )

    response = co.chat(
        model="command-r7b-12-2024",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.0
    )
    
    return response.message.content[0].text