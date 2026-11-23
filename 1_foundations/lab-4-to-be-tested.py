"""
Example 2: Multiple tools, the model picks.

We give the model four tools and ask it a range of questions. For each one it
decides whether to call a tool, which one(s), and with what arguments.

Setup:
    pip install openai
    export OPENAI_API_KEY="sk-..."
    python 02_multiple_tools.py
"""

import json
from datetime import date, timedelta

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# 1. Tool implementations (mocked, so no extra API keys are needed)
# ---------------------------------------------------------------------------

NOTES = [
    "Dentist appointment moved to Dec 3 at 2:30pm.",
    "Project Atlas kickoff notes: budget approved, launch in Q1.",
    "Gift ideas for Sam: board game, cookbook, wool socks.",
]

TODOS: list[dict] = []

FX_RATES = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 151.0}


def get_weather(city: str, unit: str = "celsius") -> dict:
    fake = {"chicago": 4, "paris": 9, "tokyo": 14}
    temp_c = fake.get(city.lower(), 20)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32)
    return {"city": city, "temperature": temp, "unit": unit, "conditions": "cloudy"}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = from_currency.upper(), to_currency.upper()
    if src not in FX_RATES or dst not in FX_RATES:
        return {"error": f"Unsupported currency. Supported: {sorted(FX_RATES)}"}
    converted = amount / FX_RATES[src] * FX_RATES[dst]
    return {"amount": amount, "from": src, "to": dst, "result": round(converted, 2)}


def search_notes(query: str) -> dict:
    hits = [n for n in NOTES if query.lower() in n.lower()]
    return {"query": query, "matches": hits}


def add_todo(task: str, due_date: str | None = None) -> dict:
    item = {"id": len(TODOS) + 1, "task": task, "due_date": due_date}
    TODOS.append(item)
    return {"status": "created", "todo": item}


# Name -> function lookup, so executing a call is one line.
DISPATCH = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
    "search_notes": search_notes,
    "add_todo": add_todo,
}

# ---------------------------------------------------------------------------
# 2. Tool schemas (what the model actually sees)
#    The descriptions are what the model uses to choose, so make them clear.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount of money from one currency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string", "description": "ISO code, e.g. 'USD'"},
                    "to_currency": {"type": "string", "description": "ISO code, e.g. 'EUR'"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the user's saved personal notes by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A keyword or short phrase"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Add a task to the user's to-do list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "due_date": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format. Omit if none was given.",
                    },
                },
                "required": ["task"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 3. The loop: ask, run whatever tools the model chose, ask again
# ---------------------------------------------------------------------------


def ask(question: str, max_rounds: int = 5) -> str:
    today = date.today()
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful personal assistant. Today is {today.isoformat()} "
                f"(tomorrow is {(today + timedelta(days=1)).isoformat()}). "
                "Use tools when they help; answer directly when they don't."
            ),
        },
        {"role": "user", "content": question},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # the model decides: no tool, one tool, or several
        )
        msg = response.choices[0].message

        # No tool calls means the model produced its final answer.
        if not msg.tool_calls:
            print("  tools chosen: (none)")
            return msg.content

        # Keep the assistant message (with its tool_calls) in the history.
        messages.append(msg)

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"  tool chosen : {name}({args})")

            try:
                result = DISPATCH[name](**args)
            except Exception as exc:  # never crash the loop on a tool error
                result = {"error": str(exc)}

            # Every tool call must be answered with a matching tool_call_id.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )

    return "Stopped: too many tool rounds."


# ---------------------------------------------------------------------------
# 4. Different questions -> different tool choices
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What's the weather like in Chicago? Fahrenheit please.",  # get_weather
    "How much is 250 USD in euros?",                            # convert_currency
    "What did I write down about the dentist?",                 # search_notes
    "Remind me to call Mom tomorrow.",                          # add_todo
    "What's the capital of France?",                            # no tool
    "What's the weather in Paris, and add a todo to pack an umbrella.",  # two tools
]

if __name__ == "__main__":
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")

    print("\nTodo list state:", json.dumps(TODOS, indent=2))