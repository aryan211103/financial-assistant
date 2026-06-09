from google import genai

client = genai.Client(
    vertexai=True,
    project="financial-assistant-498905",
    location="us-central1",
)
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence.",
)
print(resp.text)