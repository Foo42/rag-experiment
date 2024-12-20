import sys
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pathlib import Path
from typing import Optional
import numpy as np
import base64


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

    def store(self, key: str, content: str):
        if content == "":
            return
        embedding = self.embedding_of(content)
        output_file_path = self._key_to_store_path(key)
        with open(output_file_path, "w") as out:
            out.write("\n".join([str(f) for f in embedding]))

    def get_closest_n(self, target: str, n: int) -> list[tuple[str, float]]:
        target_embedding = np.array(self.embedding_of(target))
        all_files = [file for file in self.store_directory.rglob("*") if file.is_file]
        embeddings = [(file, self._load_embedding(file)) for file in all_files]
        scored = [
            (file, self._cosign_similarity(target_embedding, embedding))
            for file, embedding in embeddings
        ]
        scored.sort(reverse=True, key=lambda tup: tup[1])
        return [(self._store_path_to_key(file), score) for file, score in scored[0:n]]

    def _key_to_store_path(self, key: str) -> Path:
        output_file_name = base64.urlsafe_b64encode(key.encode()).decode()
        return self.store_directory.joinpath(output_file_name)

    def _store_path_to_key(self, store_path: Path) -> str:
        relative = store_path.relative_to(self.store_directory)
        return base64.urlsafe_b64decode(str(relative)).decode()

    def _load_embedding(self, file: Path):
        with open(file, "r") as f:
            lines = f.readlines()
        return np.array([float(line) for line in lines if line.strip() != ""])

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

    def _load_as_rag_source(self, file: Path) -> ChatCompletionMessageParam:
        with open(file, "r") as f:
            content = f.read()
        message = f"In answering questions, the following content from a file may be useful:\n {content}"
        return self._make_user_message(message)

    def add_message(self, message: str) -> str:
        if message.endswith("?") and self._db is not None:
            useful_files = [
                Path(name)
                for name, similarity in self._db.get_closest_n(message, 3)
                if similarity >= 0.5
            ]
            rag_messages = [self._load_as_rag_source(file) for file in useful_files]
            print(f"Adding context: {[file.name for file in useful_files]}")
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
    return [
        str(file_path.resolve())
        for file_path in root.rglob(pattern)
        if file_path.is_file()
    ]


def store_notes(store: EmbeddingStore, note_directory: Path):
    files = find_files(note_directory)
    for file in files:
        with open(file, "r") as f:
            content = f.read()
        store.store(str(file), content)


def get_n_closest_notes(
    store: EmbeddingStore, target: str, n: int, threshold: float = 0.5
) -> list[Path]:
    matches = store.get_closest_n(target, n)
    return [Path(file) for file, distance in matches if distance >= threshold]


def initiate_chat(chat: Chat):
    def input_generator():
        """A generator that yields user input lines until 'stop' is entered."""
        while True:
            user_input = input("You say: ")
            if user_input.lower() == "stop":
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
            paths = [str(path) for path in get_n_closest_notes(db, text, int(n))]
            for path in paths:
                print(path)
        case ["chat"]:
            initiate_chat(
                Chat(
                    client,
                    "llama-3.2-3b-instruct",
                    "You are a helpful professional who treats content from files as a useful reference when answering questions. When replying try not to include too much extraneous content",
                    db,
                )
            )
        case _:
            print("unrecognised command")


if __name__ == "__main__":
    main()
