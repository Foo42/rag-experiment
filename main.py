import sys
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pathlib import Path
from typing import Annotated, Literal, Optional, NamedTuple, TypeAlias
import numpy as np
import base64
from datetime import datetime
import hashlib

from pydantic import BaseModel, Field


class Section(NamedTuple):
    header: str
    lines: list[str]


Vector: TypeAlias = list[float]


class MarkdownChunk(BaseModel):
    source_type: Literal["markdown_section"] = "markdown_section"
    key: str
    original_file_path: str
    content: str


class EmbeddedChunk(BaseModel):
    chunk: MarkdownChunk
    vector: Vector


class MarkdownFile:
    def __init__(self, path: Path):
        lines = []
        with open(path, "r") as f:
            lines = f.readlines()
        self.title: Optional[str] = None
        self.pre_section: Optional[list[str]] = None
        self.sections: list[Section] = []

        current_section: Optional[Section] = None

        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                self.title = line[len("# ") :]
            elif line.startswith("## "):
                header = line[len("## ") :]
                if current_section is not None:
                    self.sections.append(current_section)
                current_section = Section(header=header, lines=[])
            elif current_section is None:
                if self.pre_section is None:
                    self.pre_section = []
                self.pre_section.append(line)
            elif current_section is not None:
                current_section.lines.append(line)
        if current_section is not None:
            self.sections.append(current_section)


class EmbeddingStore:
    def __init__(self, model_client: OpenAI, store_directory: Path):
        self.client = model_client
        self.store_directory = store_directory

    def embedding_of(self, content: str) -> list[float]:
        preprocessed = content.replace("\n", " ")
        return (
            self.client.embeddings.create(
                input=[preprocessed], model="text-embedding-nomic-embed-text-v1.5"
            )
            .data[0]
            .embedding
        )

    def store(self, chunk: MarkdownChunk):
        embedding = EmbeddedChunk(chunk=chunk, vector=self.embedding_of(chunk.content))
        output_file_path = self._key_to_store_path(chunk.key)
        with open(output_file_path, "w") as out:
            out.write(embedding.model_dump_json())

    def get_closest_n(self, target: str, n: int) -> list[tuple[EmbeddedChunk, float]]:
        target_embedding = np.array(self.embedding_of(target))
        all_files = [file for file in self.store_directory.rglob("*") if file.is_file]
        embeddings = [self._load_embedding(file) for file in all_files]
        scored = [
            (
                embedding,
                self._cosign_similarity(target_embedding, np.array(embedding.vector)),
            )
            for embedding in embeddings
        ]
        scored.sort(reverse=True, key=lambda tup: tup[1])
        return scored[0:n]

    def _key_to_store_path(self, key: str) -> Path:
        output_file_name = base64.urlsafe_b64encode(key.encode()).decode()
        return self.store_directory.joinpath(output_file_name)

    def _store_path_to_key(self, store_path: Path) -> str:
        relative = store_path.relative_to(self.store_directory)
        return base64.urlsafe_b64decode(str(relative)).decode()

    def _load_embedding(self, file: Path):
        with open(file, "r") as f:
            return EmbeddedChunk.model_validate_json(f.read())

    def _cosign_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if len(v1) != len(v2):
            raise ValueError("The input vectors must be of equal length.")

        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0

        cos_sim = dot_product / (magnitude_v1 * magnitude_v2)
        return float(cos_sim)


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


def find_files(root: Path, pattern="*.md") -> list[str]:
    ignore_pattern = {"node_modules", "site-packages", "gems", "Pods"}
    return [
        str(file_path.resolve())
        for file_path in root.rglob(pattern)
        if file_path.is_file() and (set(file_path.parts).isdisjoint(ignore_pattern))
    ]


def store_notes(store: EmbeddingStore, note_directory: Path):
    files = find_files(note_directory)
    for file in files:
        print("Processing: ", file)
        chunks = chunk_markdown_file(Path(file))
        for chunk in chunks:
            store.store(chunk)


def get_n_closest_notes(
    store: EmbeddingStore, target: str, n: int, threshold: float = 0.5
) -> list[tuple[MarkdownChunk, float]]:
    return [
        (stored.chunk, score)
        for stored, score in store.get_closest_n(target, n)
        if score >= threshold
    ]


def initiate_chat(chat: Chat, initial_message: Optional[str] = None):
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


def chunk_markdown_file(path: Path) -> list[MarkdownChunk]:
    md = MarkdownFile(path)
    local_file_name = path.name.split(".")[0]
    hashed_path = base64.urlsafe_b64encode(
        hashlib.md5(str(path).encode()).digest()
    ).decode()
    chunks = [
        MarkdownChunk(
            original_file_path=str(path),
            key=f"{hashed_path}##{section.header}",
            content="\n".join([local_file_name, section.header, *section.lines]),
        )
        for section in md.sections
        if len([line for line in section.lines if line != ""]) > 0
    ]
    if md.pre_section is not None and len(
        [line for line in md.pre_section if line != ""]
    ):
        chunks.append(
            MarkdownChunk(
                original_file_path=str(path),
                key=f"{hashed_path}-pre",
                content=" ".join(md.pre_section),
            )
        )

    return chunks


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
                Chat(
                    client,
                    "Hermes-3-Llama-3.1-8B-GGUF",
                    f"You are a helpful professional (named Baldric) who treats content from files as a useful reference when answering questions. When replying try not to include too much extraneous content. Today's date is {datetime.now().strftime('%B %d, %Y')}",
                    db,
                ),
                initial_message,
            )
        case _:
            print("unrecognised command")


if __name__ == "__main__":
    main()
