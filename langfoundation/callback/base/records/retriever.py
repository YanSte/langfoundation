from __future__ import annotations

from enum import auto
from typing import List, Optional

from langchain_core.documents import Document
from strenum import LowercaseStrEnum

from langfoundation.callback.base.records.base import BaseRecord


# Retriever
# ---


class RetrieverState(LowercaseStrEnum):
    START = auto()
    END = auto()


class RetrieverRecord(BaseRecord):
    """
    Represents a retriever record.

    Attributes:
        run_id (UUID): The ID of the run.
        parent_run_id (Optional[UUID], optional): The ID of the parent run. Defaults to None.
        query (str): The query string.
        state (RetrieverState): The state of the retriever.
        documents (Optional[List[Document]], optional): The list of documents. Defaults to None.
    """

    query: str
    state: RetrieverState
    documents: Optional[List[Document]] = None
