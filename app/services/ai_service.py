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
    
    system_instruction = (
        "You are an expert technical and business automation assistant for Youverify.\n"
        "Answer the user's question accurately using ONLY the verified context text provided below.\n"
        "If the answer cannot be confidently derived from the context, say: 'I am sorry, I do not have "
        "that specific detail on hand. Let me loop in a team member to assist you.' Do not invent info.\n\n"
        f"VERIFIED CONTEXT:\n{joined_context}"
    )

    response = co.chat(
        model="command-r7b-12-2024", 
        messages=[{"role": "user", "content": question}],
        system_instruction=system_instruction,
        temperature=0.2
    )
    
    return response.message.content[0].text