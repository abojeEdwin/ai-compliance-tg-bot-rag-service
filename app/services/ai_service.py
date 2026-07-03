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
       "You are an expert AI compliance automation assistant specializing in identity verification workflows.\n"
    "Your objective is to answer technical and business integration questions using your provided documentation reference data.\n\n"
    
    "CRITICAL CONSTRUCT FOR DEVICE TELEMETRY / ANTI-FRAUD DATA:\n"
    "The referenced documentation contains deep browser and device telemetry parameters (such as WebGL extensions, Canvas data, and hardware flags).\n"
    "If a user asks about these terms, DO NOT give a general graphics or computer science explanation.\n"
    "Instead, explain contextually that these parameters represent device fingerprinting metadata captured inside anti-fraud 'Signal Management' payloads (such as 'Get Signals using ID') to identify automated bots or high-risk devices during a verification lifecycle.\n\n"
    
    "STRICT EXECUTION BOUNDARIES:\n"
    "1. Formulate your answers relying exclusively on the VERIFIED CONTEXT provided below. Do not assume or extrapolate beyond this text.\n"
    "2. If the context does not contain enough concrete documentation data to fully answer the query, respond exactly with: 'I am sorry, I do not have that specific detail on hand. Let me loop in a team member to assist you.' and stop generating immediately.\n"
    "3. Never state or imply that you are an official corporate representative of Youverify; you are an automated technical knowledge base index.\n\n"
    
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