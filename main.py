import sys
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pathlib import Path
from typing import Optional, Protocol
from datetime import datetime

from pydantic_ai.messages import ModelMessage
from embeddings import EmbeddingStore, MarkdownChunk

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class ChatLike(Protocol):
    def add_message(self, message: str) -> str: ...


class PydanticChat:
    main_agent: Agent
    all_messages: list[ModelMessage]

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
    ):
        self.all_messages = []
        ollama_model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url="http://localhost:1234/v1"),
        )
        self.main_agent = Agent(
            ollama_model,
            system_prompt=system_prompt,
        )

    def add_message(self, message: str) -> str:
        result = self.main_agent.run_sync(message, message_history=self.all_messages)
        self.all_messages = result.all_messages()
        return str(result.data)


class Chat:
    def __init__(
        self,
        model_client: OpenAI,
        model_name: str,
        system_prompt: Optional[str] = None,
        embedding_store: Optional[EmbeddingStore] = None,
    ):
        self._client = model_client
        self.model_name = model_name
        self.conversation: list[ChatCompletionMessageParam] = []
        self._db = embedding_store
        if system_prompt is not None:
            self.conversation.append(self._make_system_message(system_prompt))

    def _make_system_message(self, content: str) -> ChatCompletionMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content=content)

    def _make_user_message(self, content: str) -> ChatCompletionMessageParam:
        return ChatCompletionUserMessageParam(role="user", content=content)

    def _make_assistant_message(self, content: str) -> ChatCompletionMessageParam:
        return ChatCompletionAssistantMessageParam(role="assistant", content=content)

    def _as_rag_source(self, chunk: MarkdownChunk) -> ChatCompletionMessageParam:
        message = f"In answering questions, the following content from a file may be useful:\n {chunk.content}"
        return self._make_user_message(message)

    def add_message(self, message: str) -> str:
        if message.endswith("?") and self._db is not None:
            useful_chunks = get_n_closest_notes(self._db, message, 4, 0.6)
            rag_messages = [
                self._as_rag_source(chunk) for chunk, _score in useful_chunks
            ]
            print("Adding context:")
            print(
                "\n".join(
                    [
                        f"{score}: {chunk.original_file_path}"
                        for chunk, score in useful_chunks
                    ]
                )
            )
            for rag_message in rag_messages:
                self.conversation.append(rag_message)

        self.conversation.append(self._make_user_message(message))

        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation,
            temperature=0.8,
        )

        new_message = completion.choices[0].message
        assert new_message.content is not None
        self.conversation.append(self._make_assistant_message(new_message.content))
        return new_message.content


def find_files(root: Path, pattern="*.md") -> list[Path]:
    ignore_pattern = {"node_modules", "site-packages", "gems", "Pods"}
    return [
        file_path.resolve()
        for file_path in root.rglob(pattern)
        if file_path.is_file() and (set(file_path.parts).isdisjoint(ignore_pattern))
    ]


def store_notes(store: EmbeddingStore, note_directory: Path):
    files = find_files(note_directory)
    for file in files:
        store.store_markdown_file(file)


def get_n_closest_notes(
    store: EmbeddingStore, target: str, n: int, threshold: float = 0.5
) -> list[tuple[MarkdownChunk, float]]:
    return [
        (stored.chunk, score)
        for stored, score in store.get_closest_n(target, n)
        if score >= threshold
    ]


def initiate_chat(chat: ChatLike, initial_message: Optional[str] = None):
    def input_generator():
        """A generator that yields user input lines until 'stop' is entered."""
        if initial_message is not None:
            yield initial_message
        while True:
            user_input = input("You say: ")
            if user_input.lower() in {
                "stop",
                "thanks",
                "that's all",
                "dismissed",
                "exit",
            }:
                break
            yield user_input

    for user_input in input_generator():
        reply = chat.add_message(user_input)
        print(f"Reply: {reply}")


def main():
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    db = EmbeddingStore(client, Path("/tmp/embeddings"))
    args = sys.argv[1:]
    match args:
        case ["store-all", dir]:
            store_notes(db, Path(dir))
        case ["find", n, text]:
            chunks = get_n_closest_notes(db, text, int(n))
            for chunk, score in chunks:
                print(chunk)
                print()
        case ["chat", *rest]:
            initial_message = " ".join(rest) if len(rest) > 0 else None
            initiate_chat(
                PydanticChat(
                    model_name="Hermes-3-Llama-3.1-8B-GGUF",
                    system_prompt=f"You are a helpful professional (named Baldric) who treats content from files as a useful reference when answering questions. When replying try not to include too much extraneous content. Today's date is {datetime.now().strftime('%B %d, %Y')}",
                ),
                initial_message,
            )
        case _:
            print("unrecognised command")


if __name__ == "__main__":
    main()
