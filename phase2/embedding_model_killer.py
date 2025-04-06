from utils import EmbeddingResultFileManager
import csv
import os
from constants import DATA_DIR
from pathlib import Path

result_file_manager = EmbeddingResultFileManager()

with open(result_file_manager.file_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if (
            float(row["self_similarity"]) < 0.71
            or float(row["second_similarity"]) < 0.8
        ):
            model_path = Path(
                DATA_DIR,
                f"embedding_model_{row['vector_size']}_{row['window']}_{row['min_count']}_{row['epochs']}_{row['hs']}_{row['negative']}_{row['sample']}.d2v",
            )
            vector_model_path = Path(
                DATA_DIR,
                f"embedding_model_{row['vector_size']}_{row['window']}_{row['min_count']}_{row['epochs']}_{row['hs']}_{row['negative']}_{row['sample']}.d2v.dv.vectors.npy",
            )
            wv_model_path = Path(
                DATA_DIR,
                f"embedding_model_{row['vector_size']}_{row['window']}_{row['min_count']}_{row['epochs']}_{row['hs']}_{row['negative']}_{row['sample']}.d2v.wv.vectors.npy",
            )
            syn1_model_path = Path(
                DATA_DIR,
                f"embedding_model_{row['vector_size']}_{row['window']}_{row['min_count']}_{row['epochs']}_{row['hs']}_{row['negative']}_{row['sample']}.d2v.syn1.npy",
            )

            if model_path.exists():
                # print(model_path)
                os.remove(model_path)
            if vector_model_path.exists():
                # print(vector_model_path)
                os.remove(vector_model_path)
            if syn1_model_path.exists():
                # print(syn1_model_path)
                os.remove(syn1_model_path)
            if wv_model_path.exists():
                # print(wv_model_path)
                os.remove(wv_model_path)
