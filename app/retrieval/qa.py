import os
from openai import OpenAI

def answer_question(question: str, context_chunks):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the user's question using only the context below.
If the answer is not in the context, say "I do not know based on the provided context."

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=200
    )

    return response.output_text