import numpy as np
import faiss
import pandas as pd
import json
import os
import utils.constants as const


def build_faiss_index(
    df: pd.DataFrame,
    csv_filename: str,
    dimension: int = 3072,
    existing_index: faiss.Index = None,
):
    """
    Build a FAISS index from embeddings and save metadata.

    Args:
        df: DataFrame containing embeddings and metadata
        csv_filename: Name of the CSV file (used for metadata JSON filename)
        dimension: Dimension of embeddings (default 3072 for OpenAI embeddings)
    """
    # 1. Convert embeddings to numpy array and build FAISS index
    # df["Embeddings"] = pd.to_numeric(df["Embeddings"], errors='coerce')
    # print("okok")
    vectors = np.array(df["Embeddings"].tolist())
    normalised_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

    # Use existing index or create new one
    if existing_index is not None:
        index = existing_index
    else:
        index = faiss.IndexFlatIP(dimension)
    index.add(normalised_vectors)

    # 2. Create and save metadata store
    metadata_store = []
    file_name = os.path.splitext(csv_filename)[0]

    for idx, row in df.iterrows():
        metadata = {
            "pageNumber": row.get("PageNumber", ""),
            "text": row.get("PageText", ""),
            "ImagePath": row.get("ImagePath", ""),
            "GraphicIncluded": row.get("Visual_Input_Processed", False),
        }
        metadata_store.append(metadata)

    # Create metadata directory if it doesn't exist
    # metadata_dir = os.path.join(const.DOC_ROOT_PATH, "metadata")
    # os.makedirs(metadata_dir, exist_ok=True)

    # Save metadata as JSON
    json_filename = os.path.splitext(csv_filename)[0] + ".json"
    json_path = os.path.join(const.METADATA_ROOT_PATH, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=4)

    # 3. Save FAISS index
    # index_dir = os.path.join(const.DOC_ROOT_PATH, "index")
    # os.makedirs(index_dir, exist_ok=True)

    index_filename = os.path.splitext(csv_filename)[0] + ".index"
    index_path = os.path.join(const.INDEX_ROOT_PATH, index_filename)
    faiss.write_index(index, index_path)

    return {
        "index_path": index_path,
        "metadata_path": json_path,
        "num_vectors": len(normalised_vectors),
    }
