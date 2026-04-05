import json
import requests
from ddgs import DDGS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when the user asks about recent events, current data, or anything you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_analyzer",
            "description": "Parses, analyzes, and provides actionable feedback on a provided block of source code. Use this tool when the user needs to check for bugs, optimize performance, understand complexity, or refactor existing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_snippet": {
                        "description": "The complete source code provided by the user. This must be accurate and ready to run (or be the full file content).",
                        "type": "STRING"
                    },
                    "language": {
                        "description": "The programming language of the code (e.g., Python, JavaScript, Java, C++).",
                        "type": "STRING"
                    },
                    "analysis_focus": {
                        "description": "What specific aspect of the code needs attention. Choose one or more from: 'security vulnerability check', 'performance optimization', 'readability/style guide adherence', 'complexity assessment', or 'bug identification'.",
                        "type": "ARRAY",
                        "items": {
                            "type": "string",
                            "enum": [
                                "security vulnerability check",
                                "performance optimization",
                                "readability assessment",
                                "complexity assessment"
                            ]
                        }
                    }
                }
            }
        }
    ]

def run_chat_session():
    """Runs the main interactive chat loop."""
    print("--- Gemma Chat Session Initialized ---")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("\n--- Session Ended ---")
            break
        
        # In a real application, this is where the API call would happen.
        # For this example, we simulate the response structure.
        
        # Simulate API call and response handling
        try:
            # Placeholder for actual API call logic
            print("\n[Simulating API call to Gemma model...]")
            
            # Simple simulation of a response
            if "hello" in user_input.lower():
                response = "Hello! I am a large language model, ready to assist you today. How can I help?"
            elif "weather" in user_input.lower():
                response = "I cannot check the live weather, but I recommend checking a dedicated weather service!"
            else:
                response = f"I received your message: '{user_input}'. I am processing this request using my advanced language capabilities."

            print(f"\nAI: {response}")

        except Exception as e:
            print(f"\nAI Error: An error occurred during processing: {e}")


def main():
    """Main entry point."""
    run_chat_session()

if __name__ == "__main__":
    main()