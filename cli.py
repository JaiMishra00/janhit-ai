from pprint import pprint
import uuid
from pathlib import Path

from graph import app, qdrant_client
from utils.memory_store import store_memory, retrieve_memory
from Agents.embedding_agent import embedder

# ---------- SETUP ----------
SESSION_ID = str(uuid.uuid4())
turn_id = 0

print(f"\nLegal RAG CLI with Memory")
print(f"[Session ID] {SESSION_ID}\n")
print("Attach files using: file:<path> <your question>\n")


def parse_input(user_input: str):
    """
    Parses:
    file:path/to/file.pdf question
    OR
    question only
    """
    file_path = None
    query = user_input

    if user_input.startswith("file:"):
        parts = user_input.split(maxsplit=1)
        raw_path = parts[0].replace("file:", "")
        file_path = Path(raw_path).as_posix()
        query = parts[1] if len(parts) > 1 else ""

    return query.strip(), file_path


def chat():
    global turn_id

    while True:
        user_input = input("> ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting")
            break

        query, file_path = parse_input(user_input)

        # -------- RETRIEVE MEMORY --------
        memory_context = retrieve_memory(
            client=qdrant_client,
            embedder=embedder,
            session_id=SESSION_ID,
            current_query=query,
            top_k=4
        )

        # -------- BUILD STATE --------
        state = {
        "query": query,
        "files": [file_path] if file_path else [],  # ALWAYS list
        "session_id": SESSION_ID,
        "relevant_memory": memory_context,
        "embedder": embedder,
        "qdrant_client": qdrant_client
    }


        result = app.invoke(state)
        response = result.get("final_response")

        if response:
            print("\n" + response + "\n")

            # -------- STORE MEMORY --------
            store_memory(
                client=qdrant_client,
                embedder=embedder,
                session_id=SESSION_ID,
                role="user",
                text=user_input,
                turn_id=turn_id
            )

            store_memory(
                client=qdrant_client,
                embedder=embedder,
                session_id=SESSION_ID,
                role="assistant",
                text=response,
                turn_id=turn_id
            )

            turn_id += 1
        else:
            print("\nNo response generated\n")


if __name__ == "__main__":
    chat()
