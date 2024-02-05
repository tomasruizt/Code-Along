from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an great professor in computer science and are mentoring an full-time programmer who wants to start a PhD."},
    {"role": "user", "content": "Tell me what is important to find a good PhD position."},
  ]
)

print(completion.choices[0].message)
