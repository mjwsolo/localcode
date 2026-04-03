import json
import requests
from ddgs import DDGS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

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
    }
]


def search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"**{r['title']}**\n{r['body']}\n{r['href']}\n")
    return "\n".join(results) if results else "No results found."


def chat(messages):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False
    })
    resp.raise_for_status()
    return resp.json()["message"]


def main():
    print(f"\n  Gemma 4 Chat (with web search)")
    print(f"  Model: {MODEL}")
    print(f"  Type 'quit' to exit\n")

    messages = []

    while True:
        try:
            user_input = input("\033[1m> \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = chat(messages)
        messages.append(response)

        # Handle tool calls
        while response.get("tool_calls"):
            for tool_call in response["tool_calls"]:
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]

                if name == "web_search":
                    print(f"\033[90m  Searching: {args['query']}\033[0m")
                    result = search(args["query"])
                else:
                    result = f"Unknown tool: {name}"

                messages.append({
                    "role": "tool",
                    "content": result
                })

            response = chat(messages)
            messages.append(response)

        if response.get("content"):
            print(f"\n{response['content']}\n")


if __name__ == "__main__":
    main()
