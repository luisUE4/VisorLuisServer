import gc
import os
import time

import torch
from sentence_transformers import SentenceTransformer

from .config import (
    MODEL_IDLE_SECONDS,
    MODEL_PATH,
)


# =============================================================
# HUGGING FACE / TRANSFORMERS OFFLINE
# =============================================================
#
# El modelo debe existir previamente en MODEL_PATH.
#
# Estas variables impiden que Hugging Face / Transformers
# intenten acceder a Internet durante la ejecución.
# =============================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


_model = None
_last_use = 0.0


# =============================================================
# CARGAR MODELO
# =============================================================

def get_model():

    global _model
    global _last_use

    if _model is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[RAG] Cargando modelo local "
            f"desde: {MODEL_PATH}"
        )

        print(
            f"[RAG] Dispositivo: {device}"
        )

        # -----------------------------------------------------
        # Verificar que el modelo local existe.
        #
        # Si no existe, fallamos inmediatamente.
        # NO se intenta descargar desde Internet.
        # -----------------------------------------------------

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                "No existe el modelo local de embeddings:\n"
                f"{MODEL_PATH}\n\n"
                "El modelo debe descargarse previamente "
                "durante la instalación del servidor."
            )

        if not MODEL_PATH.is_dir():

            raise RuntimeError(
                f"MODEL_PATH no es un directorio: "
                f"{MODEL_PATH}"
            )

        # -----------------------------------------------------
        # Cargar exclusivamente desde disco local.
        # -----------------------------------------------------

        _model = SentenceTransformer(
            str(MODEL_PATH),
            device=device,
            local_files_only=True,
        )

        print(
            f"[RAG] Modelo local listo: "
            f"{_model.device}"
        )

    _last_use = time.time()

    return _model


# =============================================================
# GENERAR EMBEDDINGS DE DOCUMENTOS
# =============================================================

def encode_passages(texts):

    model = get_model()

    prefixed = [
        "passage: " + text
        for text in texts
    ]

    return model.encode(
        prefixed,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


# =============================================================
# GENERAR EMBEDDING DE CONSULTA
# =============================================================

def encode_query(query):

    model = get_model()

    return model.encode(
        "query: " + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# =============================================================
# LIBERAR MODELO SI ESTÁ INACTIVO
# =============================================================

def release_if_idle():

    global _model

    if _model is None:
        return

    if time.time() - _last_use < MODEL_IDLE_SECONDS:
        return

    print(
        "[RAG] Modelo inactivo. "
        "Liberando GPU..."
    )

    del _model

    _model = None

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(
        "[RAG] GPU liberada."
    )