import google.generativeai as genai
API_KEY = ""
genai.configure(api_key=API_KEY)
model= genai.GenerativeModel("gemini-2.0-flash")
chat= model.start_chat()

response= chat.send_message("who is elle woods in short")
print("ai:",response.text)