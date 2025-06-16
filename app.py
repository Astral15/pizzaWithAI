from dataclasses import asdict, dataclass
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gemini-1.5-flash")

SYSTEM_PROMPT = """
You are a pizza ordering bot for a pizzeria.

Your ONLY job is to take pizza orders, ask relevant questions about the order (e.g., size, toppings, crust type, delivery or pickup), and confirm the final order.

- Do NOT talk about anything unrelated to pizza, food, or ordering.  
- Do NOT answer questions outside your role (e.g., philosophy, weather, current events).  
- Never break character as a pizza bot.

- Always respond as a friendly and efficient pizza ordering assistant.  
- Keep your responses short and focused on completing the pizza order.  
- If a user says something off-topic, politely guide them back to placing an order.

Respond clearly and concisely, using no more than 2–3 sentences per reply.

You are not a general assistant. You are not allowed to engage in casual conversation. Only pizza.
"""

MODEL_RESPONSE = "I understand. I am a pizza bot. I will only take pizza orders."


@dataclass
class Message:
    role: str
    parts: list[str]
    Model = "model"
    User = "user"

    def __init__(self, role: str, content: str):
        self.role = role
        self.parts = [content]


class GeminiService:
    generation_config = {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 20,
        'max_output_tokens': 512,
        'stop_sequences': [],
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    def __init__(self):
        try:
            self.model = genai.GenerativeModel(
                model_name=CHAT_MODEL_NAME,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            # Warm-up request (optional, helps reduce first-response lag)
            self.model.start_chat(history=[{"role": "user", "parts": ["hello"]}])
        except Exception as e:
            print(f"❌ Error initializing model {CHAT_MODEL_NAME}: {e}")
            raise

    def get_completion(self, messages: list[Message], message: str) -> str:
        messages_dict = [asdict(m) for m in messages]
        convo = self.model.start_chat(history=messages_dict)

        response_stream = convo.send_message(message, stream=True)

        response_text = ""
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                response_text += chunk.text

        print() 
        return response_text


class ChattyUI:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT, system_prompt_response: str = MODEL_RESPONSE):
        self.service = GeminiService()
        self.system_prompt = system_prompt
        self.system_prompt_response = system_prompt_response
        self.chat_history = []

    def answer(self, message: str) -> str:
        messages = [Message(Message.User, self.system_prompt),
                    Message(Message.Model, self.system_prompt_response)]

        for user_msg, bot_msg in self.chat_history:
            messages.append(Message(Message.User, user_msg))
            messages.append(Message(Message.Model, bot_msg))

        response = self.service.get_completion(messages=messages, message=message)

        self.chat_history.append((message, response))
        return response

    def start_chat(self):
        print("🍕 Welcome to Pizza Bot! 🍕")
        print("Type your pizza order or questions. Type 'quit' to exit.")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye', 'stop']:
                    print("🍕 Thanks for visiting Pizza Bot! Goodbye!")
                    break
                if not user_input:
                    print("Please enter something!")
                    continue

                print("🤖 Pizza Bot: ", end="", flush=True)
                self.answer(user_input)

            except KeyboardInterrupt:
                print("\n\n🍕 Pizza Bot session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

    def get_chat_history(self):
        return self.chat_history

    def clear_history(self):
        self.chat_history = []
        print("Chat history cleared!")


if __name__ == "__main__":
    pizza_bot = ChattyUI()
    pizza_bot.start_chat()

    print("\n" + "="*50)
    print("FINAL CHAT HISTORY:")
    for i, (user_msg, bot_msg) in enumerate(pizza_bot.get_chat_history(), 1):
        print(f"{i}. User: {user_msg}")
        print(f"   Bot: {bot_msg}")
