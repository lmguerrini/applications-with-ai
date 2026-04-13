from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


Topic = Literal["langchain", "rag", "chroma", "streamlit", "tool_calling", "prompting"]
Library = Literal["langchain", "chroma", "streamlit", "openai", "general"]
DocType = Literal["concept", "how_to", "example", "troubleshooting"]
Difficulty = Literal["intro", "intermediate", "advanced"]
ErrorFamily = Literal["imports", "api", "retrieval", "ui", "persistence"]
ToolName = Literal[
    "estimate_openai_cost",
    "diagnose_stack_error",
    "recommend_retrieval_config",
]
SupportedToolLibrary = Literal["langchain", "chroma", "streamlit", "openai"]
RetrievalTaskType = Literal["question_answering", "debugging", "implementation"]
OfficialDocsProviderMode = Literal["official_mcp", "official_fallback"]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: str
    source_path: str
    title: str
    topic: Topic
    library: Library
    doc_type: DocType
    difficulty: Difficulty
    error_family: ErrorFamily | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value:
            raise ValueError("Document metadata must include a non-empty title.")
        return value


class ChunkMetadata(DocumentMetadata):
    chunk_index: int


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 3

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Retrieval query must not be empty.")
        return cleaned

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value < 1 or value > 10:
            raise ValueError("top_k must be between 1 and 10.")
        return value


class RetrievalFilters(BaseModel):
    topic: Topic | None = None
    library: Library | None = None
    doc_type: DocType | None = None
    error_family: ErrorFamily | None = None

    def as_chroma_filter(self) -> dict[str, object]:
        filters = {
            key: value
            for key, value in self.model_dump().items()
            if value is not None
        }
        if not filters:
            return {}
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: value}
        return {
            "$and": [{key: value} for key, value in filters.items()]
        }


class RetrievedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata


class RetrievalResult(BaseModel):
    rewritten_query: str
    applied_filters: RetrievalFilters
    used_fallback: bool
    chunks: list[RetrievedChunk]
    sources: list[str]


class EstimateOpenAICostInput(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    num_calls: int = 1

    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def validate_token_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Token counts must be zero or greater.")
        return value

    @field_validator("num_calls")
    @classmethod
    def validate_num_calls(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_calls must be at least 1.")
        return value


class EstimateOpenAICostOutput(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    num_calls: int
    estimated_input_cost_usd: float
    estimated_output_cost_usd: float
    estimated_total_cost_usd: float


class DiagnoseStackErrorInput(BaseModel):
    library: SupportedToolLibrary
    error_message: str
    code_context_summary: str | None = None

    @field_validator("error_message")
    @classmethod
    def validate_error_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("error_message must not be empty.")
        return cleaned


class DiagnoseStackErrorOutput(BaseModel):
    library: SupportedToolLibrary
    error_category: str
    likely_causes: list[str]
    recommended_checks: list[str]


class RecommendRetrievalConfigInput(BaseModel):
    content_type: DocType
    document_length: Difficulty
    task_type: RetrievalTaskType


class RecommendRetrievalConfigOutput(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_metadata_filters: bool
    rationale: str


class ToolInvocationResult(BaseModel):
    tool_name: ToolName
    raw_query: str
    tool_input: (
        EstimateOpenAICostInput
        | DiagnoseStackErrorInput
        | RecommendRetrievalConfigInput
        | None
    ) = None
    tool_output: (
        EstimateOpenAICostOutput
        | DiagnoseStackErrorOutput
        | RecommendRetrievalConfigOutput
        | None
    ) = None
    tool_error: str | None = None


class RequestUsage(BaseModel):
    model_name: str | None = None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None = None

    @field_validator("input_tokens", "output_tokens", "total_tokens")
    @classmethod
    def validate_token_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Token counts must be zero or greater.")
        return value


class OfficialDocsLookupRequest(BaseModel):
    query: str
    library: SupportedToolLibrary

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Official docs query must not be empty.")
        return cleaned


class OfficialDocsSnippet(BaseModel):
    text: str
    rank: int | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Official docs snippets must include non-empty text.")
        return cleaned

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("Official docs snippet rank must be at least 1.")
        return value


class OfficialDocsDocument(BaseModel):
    title: str
    url: str
    provider_mode: OfficialDocsProviderMode
    snippets: list[OfficialDocsSnippet]

    @field_validator("title", "url")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Official docs documents must include non-empty text fields.")
        return cleaned

    @field_validator("snippets")
    @classmethod
    def validate_snippets(cls, value: list[OfficialDocsSnippet]) -> list[OfficialDocsSnippet]:
        if not value:
            raise ValueError("Official docs documents must include at least one snippet.")
        return value


class OfficialDocsLookupResult(BaseModel):
    library: SupportedToolLibrary
    documents: list[OfficialDocsDocument]


class OfficialDocsAnswerResult(BaseModel):
    library: SupportedToolLibrary
    answer: str
    lookup_result: OfficialDocsLookupResult
    usage: RequestUsage | None = None

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Official docs answer must not be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_library_alignment(self):
        if self.lookup_result.library != self.library:
            raise ValueError(
                "Official docs answer result library must match the lookup result library."
            )
        return self


class AnswerResult(BaseModel):
    answer: str
    used_context: bool
    retrieval: RetrievalResult | None
    answer_sources: list[str]
    tool_result: ToolInvocationResult | None = None
    official_docs_result: OfficialDocsAnswerResult | None = None
    usage: RequestUsage | None = None
