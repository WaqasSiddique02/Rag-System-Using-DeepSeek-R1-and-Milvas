from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from dotenv import load_dotenv
import os

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "documents"

def connect_to_milvus():
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_TOKEN
    )

def create_collection(dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="Document embeddings")

    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    print(f"✅ Collection '{COLLECTION_NAME}' created and indexed.")

    return collection


def get_or_create_collection(dim):
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(name=COLLECTION_NAME)

        # ✅ Check if index exists on 'embedding'
        index_info = collection.indexes
        if not index_info:
            index_params = {
                "index_type": "IVF_FLAT",   # or "HNSW"
                "metric_type": "L2",        # or "COSINE"
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            print(f"✅ Index created on existing collection '{COLLECTION_NAME}'")
    else:
        collection = create_collection(dim)

    # ✅ Safe to load now
    collection.load()
    return collection


def insert_documents(collection, texts, embeddings):
    entities = [embeddings, texts]
    collection.insert(entities)
    collection.flush()

def search(collection, query_embedding, top_k):
    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    hits = results[0]
    return [hit.entity.get("text") for hit in hits]
