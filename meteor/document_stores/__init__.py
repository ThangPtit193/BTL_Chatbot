from meteor.document_stores.base import BaseDocumentStore
from meteor.document_stores.memory import InMemoryDocumentStore
from meteor.utils.import_utils import safe_import

InMemoryDocumentStore = safe_import(
    "meteor.document_stores.memory", "InMemoryDocumentStore", "memory"
)
