import os
import cohere
from dotenv import load_dotenv

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

question = "How can i integrate the nin api"
system_instruction = (
    "You are an expert technical and business automation assistant for Youverify.\n"
    "Answer the user's question accurately using ONLY the verified context text provided below.\n"
    "If the answer cannot be confidently derived from the context, say: 'I am sorry, I do not have "
    "that specific detail on hand. Let me loop in a team member to assist you.' Do not invent info.\n\n"
    f"VERIFIED CONTEXT:\nSome random irrelevant context."
)

response = co.chat(
    model="command-r7b-12-2024",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question}
    ],
    temperature=0.2
)

print(response.message.content[0].text)
