import base64
import hashlib
from pathlib import Path
from typing import Literal, Optional, NamedTuple, TypeAlias
import numpy as np
import base64
from openai import OpenAI
from pydantic import BaseModel


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

    def store_markdown_file(self, path: Path):
        chunks = chunk_markdown_file(path)
        for chunk in chunks:
            self.store_chunk(chunk)

    def store_chunk(self, chunk: MarkdownChunk):
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
