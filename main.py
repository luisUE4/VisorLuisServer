

# ============================================================
# COMPROBAR ENTORNO VIRTUAL .venv  ANTES DE INICIAR EL SERVIDOR
# ============================================================
import sys
from utils.MixUtils import check_virtual_environment
if not check_virtual_environment():
    sys.exit(1)

# ============================================================
# imports
# ============================================================

import os
import socket
import logging
from pathlib import Path

import python_multipart # Para validar la existencia antes de que falle en tiempo de ejecución
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, status, Depends, Request, Header
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import asyncio
from mi_rag.search import search as rag_search
from utils.FileUtils import get_available_filename, calculate_file_hash
from mi_rag.database import (
    init_database,
    get_connection,
    create_document,
    list_categories,
    delete_category,
    create_category,
    get_category_by_id,
    get_category_by_name,
    update_document_category,
    get_or_create_general_category
)
import lancedb
from mi_rag.config import LANCEDB_DIR,API_KEY
import traceback
from mi_rag.sync_documents import sync_documents
from utils.FileUtils import calculate_file_hash

# =============================================================
# LOGGING
# =============================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "api.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # Consola
        logging.StreamHandler(),

        # Archivo
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
    ],
)

logger = logging.getLogger("api_logger")


# =============================================================
# Configuración básica
# =============================================================

API_KEY_NAME = "X-API-Key"
CLIENT_ID_HEADER_NAME = "X-Client-ID" # Cabecera para identificar al usuario/cliente
BASE_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mis_archivos")

# Asegurarse de que el directorio de almacenamiento exista
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

# Seguridad: API Key Header
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales de API Key no válidas o faltantes.",
    )

# Dependencia para extraer y validar el ID del cliente y proveer su directorio exclusivo
async def get_client_directory(x_client_id: str = Header(..., alias=CLIENT_ID_HEADER_NAME, description="ID único del cliente/usuario")) -> str:
    # Limpiar el client_id para evitar caracteres extraños o ataques de ruta
    safe_client_id = "".join(c for c in x_client_id if c.isalnum() or c in ('_', '-'))
    if not safe_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identificador de cliente (X-Client-ID) inválido o faltante."
        )
    
    # Crear la ruta exclusiva para este usuario: mis_archivos/cliente_id/
    user_dir = os.path.join(BASE_UPLOAD_DIR, safe_client_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


# ============================================================
# Lanzar fastapi 
# ============================================================
app = FastAPI(
    title="Servicio de Archivos Seguro para Android LAN",
    description="API que permite subir, listar y descargar archivos locales mediante HTTPS.",
    version="1.0.0"
)

# ============================================================
# Inicializar base de datos RAG
# ============================================================
init_database()

# ============================================================
# Sincronizar archivos físicos con la BD
# ============================================================
sync_documents(BASE_UPLOAD_DIR)






# Middleware para registrar cada request (endpoint ejecutado y host cliente que lo solicitó)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_host = request.client.host if request.client else "Unknown"
    client_port = request.client.port if request.client else ""
    client_id = request.headers.get(CLIENT_ID_HEADER_NAME, "No-Client-ID")
    endpoint = f"{request.method} {request.url.path}"
    
    # Log al recibir el request
    logger.info(f"Petición recibida - Endpoint: '{endpoint}' | Cliente: {client_host}:{client_port} usuario:{client_id}")
    
    response = await call_next(request)
    return response




@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Servidor de archivos HTTPS activo. Listo para conectar con la app Android.",
        "endpoints_disponibles": {
            
        }
    }



# Endpoint para Subir Archivos 
@app.post("/upload", response_class=JSONResponse)
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
    user_dir: str = Depends(get_client_directory)
):
    """
    Endpoint para Subir Archivos (optimizado para streaming)
    alamacena en base de datos en documento
    agrega a cola de procesamiento de ORC y despues indexado en bd embeddings 

    **nota importante cuando un archivo que se sube tiene el mismo nombre que uno que existe en el servidor 
    **entonces se renombra y se avisa en la respuesta con flag renombrado:true

    """
    try:
        original_filename = os.path.basename(file.filename)

        # asegura que no sobrescribamos un documento, busca un nombre que no exista 
        filename, renamed = get_available_filename(
            user_dir,
            original_filename,
        )

        file_path = os.path.join(
            user_dir,
            filename,
        )

        
        file_size = 0
        CHUNK_SIZE = 1024 * 1024  # Leer de 1 MB en 1 MB

        # Escribir el archivo en disco por fragmentos para no saturar la RAM
        with open(file_path, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                f.write(chunk)
                file_size += len(chunk)

        # generando hash
        hash = calculate_file_hash(file_path)

        logger.info(f"upload -> documento guardado  {file_path}  hash:{hash}")

        # Agregar el documento a Base de datos y a la cola de procesamiento RAG .
        document_id, job_id = create_document(
            user_id=os.path.basename(user_dir),
            filename=filename,
            filepath=file_path,
            hash_value= hash
        )
        
        return {
            "success": True,
            "original_filename": original_filename,
            "filename": filename,
            # Indica si fue necesario modificar el nombre 
            "renombrado": renamed,
            "message": f"Archivo '{filename}' subido con éxito.",
            "size": file_size,            
            "rag": {
                "document_id": document_id,
                "job_id": job_id,
                "status": "queued"
            }
        }
    except Exception as e:
        logger.error(f"Error al subir archivo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar el archivo: {str(e)}"
        )
    finally:
        await file.close() # Asegurar que se cierren los recursos del archivo temporal

# Endpoint para Listar Archivos
# responde con lista arreglo de objetos de la lista de archivos
@app.get("/list", response_model=List[dict])
async def list_files(
    api_key: str = Depends(get_api_key),
    user_dir: str = Depends(get_client_directory)
    ):
    try:
        files = []
        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            if os.path.isfile(file_path):
                stat_info = os.stat(file_path)                
                files.append({
                    "filename": filename,
                    "size": stat_info.st_size,
                    "created_at": str(datetime_to_str(stat_info.st_mtime))
                })
        
        return files
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar archivos: {str(e)}"
        )



# Endpoint para Listar Documentos con sus Hashes
@app.get("/list_con_hash", response_model=List[dict])
async def list_files_with_hash(
    api_key: str = Depends(get_api_key),
    user_dir: str = Depends(get_client_directory)
):
    try:
        user_id = os.path.basename(  os.path.normpath(user_dir)  )
        conn = get_connection()

        try:
            rows = conn.execute(
                """
                SELECT
                    d.filename,
                    d.filepath,
                    d.hash,
                    d.category_id,
                    c.name AS category_name,
                    d.created_at,
                    d.status

                FROM documents d

                LEFT JOIN categories c
                    ON c.id = d.category_id
                    AND c.user_id = d.user_id

                WHERE d.user_id = ?

                ORDER BY d.created_at ASC
                """,
                (
                    user_id,
                )
            ).fetchall()

        finally:
            conn.close()

        files = []

        for row in rows:
            files.append({
                "filename": row["filename"],
                "category_id": row["category_id"],
                "category_name": row["category_name"],
                "hash": row["hash"],
                "created_at": row["created_at"],
                "status": row["status"],
            })

            logger.info(
                "[LIST_CON_HASH] "
                "usuario=%s | archivo=%s | hash=%s | "
                "category_id=%s | category_name=%s",
                user_id,
                row["filename"],
                row["hash"],
                row["category_id"],
                row["category_name"],
            )

        logger.info(
            "[LIST_CON_HASH] Usuario=%s | Documentos encontrados=%d",
            user_id,
            len(files),
        )

        return files

    except Exception as e:
        logger.error(
            f"list_con_hash -> {str(e)}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Error al listar documentos con hashes: "
                f"{str(e)}"
            )
        )



# =============================================================
# LISTAR ARCHIVOS FÍSICOS + DETALLES DE BD
# =============================================================

@app.get("/list_details", response_class=JSONResponse)
async def list_details(
    api_key: str = Depends(get_api_key),
    user_dir: str = Depends(get_client_directory),
):
    """
    Lista los archivos PDF físicos del directorio del usuario
    y los relaciona con sus registros en SQLite.

    Permite detectar:

        - documentos físicos registrados en BD
        - documentos físicos sin registro en BD

    No modifica ningún dato.
    """

    try:
        user_id = os.path.basename(
            os.path.normpath(user_dir)
        )
        user_path = Path(user_dir).resolve()
        if not user_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="El directorio del usuario no existe.",
            )
        if not user_path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="La ruta del usuario no es un directorio.",
            )
        print("")
        print("=" * 60)
        print("[RAG] LIST_DETAILS")
        print("=" * 60)
        print( f"[RAG] Usuario: {user_id}" )
        print( f"[RAG] Directorio: {user_path}" )
        # =====================================================
        # 1. Obtener documentos registrados en SQLite
        # =====================================================
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT
                    id,
                    user_id,
                    filename,
                    filepath,
                    hash,
                    category_id,
                    ocr_filepath,
                    status,
                    error,
                    created_at,
                    processed_at
                FROM documents
                WHERE user_id = ?
                ORDER BY filename COLLATE NOCASE
                """,
                (
                    user_id,
                ),
            ).fetchall()
        finally:
            conn.close()
        documents_by_path = {}

        for row in rows:
            document = dict(row)
            db_path = str(
                Path( document["filepath"] ).resolve()
            )
            documents_by_path[
                db_path
            ] = document

        # =====================================================
        # 2. Buscar archivos físicos PDF
        # =====================================================

        physical_files = []
        for filepath in user_path.iterdir():

            if not filepath.is_file():
                continue
            if filepath.suffix.lower() != ".pdf":
                continue
            resolved_path = filepath.resolve()

            path_string = str(
                resolved_path
            )
            document = documents_by_path.get(
                path_string
            )

            # =================================================
            # 3. Información física
            # =================================================

            stat = filepath.stat()

            item = {
                "filename": filepath.name,
                "filepath": path_string,
                "document_id": (
                    document["id"]
                    if document
                    else None
                ),
                "hash": (
                    document["hash"]
                    if document
                    else None
                ),
                "registered_in_db": (
                    document is not None
                ),
                "status": (
                    document["status"]
                    if document
                    else None
                ),
                "category_id": (
                    document["category_id"]
                    if document
                    else None
                ),
                "ocr_filepath": (
                    document["ocr_filepath"]
                    if document
                    else None
                ),
                "error": (
                    document["error"]
                    if document
                    else None
                ),
                "created_at": (
                    document["created_at"]
                    if document
                    else None
                ),
                "processed_at": (
                    document["processed_at"]
                    if document
                    else None
                ),
                "file_size": stat.st_size,
                "modified_at": (
                    stat.st_mtime
                ),
            }

            physical_files.append(
                item
            )

        # =====================================================
        # 4. Ordenar por nombre
        # =====================================================

        physical_files.sort(
            key=lambda x: x["filename"].lower()
        )

        registered_count = sum(
            1
            for item in physical_files
            if item["registered_in_db"]
        )

        unregistered_count = sum(
            1
            for item in physical_files
            if not item["registered_in_db"]
        )

        # =====================================================
        # 5. Resultado
        # =====================================================

        print(
            f"[RAG] Archivos PDF físicos: "
            f"{len(physical_files)}"
        )

        print(
            f"[RAG] Registrados en BD: "
            f"{registered_count}"
        )

        print(
            f"[RAG] Sin registro en BD: "
            f"{unregistered_count}"
        )

        return {
            "success": True,
            "user_id": user_id,
            "directory": str(user_path),
            "count": len( physical_files ),
            "registered_count": ( registered_count ),
            "unregistered_count": ( unregistered_count ),
            "files": physical_files,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error en /list_details: %s",
            str(e),
        )
        logger.error(
            traceback.format_exc()
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Error al listar los documentos: "
                f"{str(e)}"
            ),
        )







class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    limit: int = 10

@app.post("/buscar")
async def buscar(
    request: SearchRequest,
    api_key: str = Depends(get_api_key),
    x_client_id: str = Header(
        ...,
        alias=CLIENT_ID_HEADER_NAME,
    ),
):

    user_id = "".join(
        c
        for c in x_client_id
        if c.isalnum() or c in ("_", "-")
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identificador de cliente inválido."
        )

    if request.mode not in (
        "exact",
        "semantic",
        "hybrid",
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mode debe ser exact, semantic o hybrid."
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La consulta no puede estar vacía."
        )

    limit = min(
        max(request.limit, 1),
        50,
    ) #cambiar el numero para aumentar cantidad de resultados entregados al cliente

    try:

        # La búsqueda semántica ejecuta inferencia
        # y puede utilizar GPU. No queremos bloquear
        # el event loop de FastAPI.
        results = await asyncio.to_thread(
            rag_search,
            user_id,
            request.query,
            request.mode,
            limit,
        )


        # =====================================================
        # RESPUESTA a cliente
        # =====================================================


        return {
            "success": True,
            "query": request.query,
            "mode": request.mode,
            "results": results,
        }

    except Exception as e:

        logger.exception(
            "Error realizando búsqueda RAG"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en búsqueda: {str(e)}",
        )

# Endpoint para Descargar Archivo
@app.get("/download/{filename}")
async def download_file(
    filename: str,
    api_key: str = Depends(get_api_key),
    user_dir: str = Depends(get_client_directory)
):
    # Evitar ataques de Directory Traversal
    filename = os.path.basename(filename)
    file_path = os.path.join(user_dir, filename)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"El archivo '{filename}' no existe en el servidor."
        )
        
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )





# =============================================================
# ELIMINAR DOCUMENTO
# =============================================================

@app.delete("/eliminar_documento", response_class=JSONResponse)
async def eliminar_documento(
    document_id: str,
    api_key: str = Depends(get_api_key),
    user_dir: str = Depends(get_client_directory),
):
    """
    Elimina completamente un documento.

    Se eliminan:

        - archivo físico
        - embeddings de LanceDB
        - registros FTS5
        - jobs asociados
        - registro documents

    Si el documento eliminado es un documento OCR:

        original.pdf
            |
            +-- ocr_filepath -> original_texto.pdf

    entonces al eliminar original_texto.pdf:

        original.pdf
            |
            +-- ocr_filepath = NULL

    IMPORTANTE:

    Si se elimina el documento original, NO se elimina
    automáticamente su documento OCR.

    También se impide eliminar un documento mientras
    tenga un job en estado "processing".
    """

    conn = None

    try:

        # =====================================================
        # IDENTIDAD DEL USUARIO
        # =====================================================

        user_id = os.path.basename(
            os.path.normpath(user_dir)
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("[RAG] SOLICITUD DE ELIMINACIÓN")
        logger.info("=" * 60)
        logger.info(
            f"[RAG] document_id: {document_id}"
        )
        logger.info(
            f"[RAG] user_id: {user_id}"
        )

        # =====================================================
        # =====================================================
        # FASE 1
        # COMPROBAR TODO ANTES DE ELIMINAR
        # =====================================================
        # =====================================================

        logger.info("")
        logger.info(
            "[RAG] Fase 1: comprobando documento..."
        )

        # =====================================================
        # 1. Obtener documento
        # =====================================================

        conn = get_connection()

        document_row = conn.execute(
            """
            SELECT
                id,
                user_id,
                filename,
                filepath,
                ocr_filepath,
                status
            FROM documents
            WHERE id = ?
            AND user_id = ?
            LIMIT 1
            """,
            (
                document_id,
                user_id,
            ),
        ).fetchone()

        conn.close()
        conn = None

        if document_row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Documento no encontrado.",
            )

        document = dict(document_row)
        filename = document["filename"]
        filepath = Path( document["filepath"] ).resolve()

        logger.info( f"[RAG] Documento encontrado: {filename}" )
        logger.info( f"[RAG] Ruta: {filepath}"  )

        # =====================================================
        # 2. Comprobar estado
        # =====================================================

        if document["status"] == "processing":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "El documento está siendo procesado "
                    "actualmente. Espere a que termine "
                    "antes de eliminarlo."
                ),
            )

        # =====================================================
        # 3. Comprobar jobs asociados
        # =====================================================

        conn = get_connection()

        job_rows = conn.execute(
            """
            SELECT
                id,
                type,
                status
            FROM jobs
            WHERE document_id = ?
            """,
            (
                document_id,
            ),
        ).fetchall()

        conn.close()
        conn = None

        jobs = [
            dict(row)
            for row in job_rows
        ]

        processing_jobs = [
            job
            for job in jobs
            if job["status"] == "processing"
        ]

        if processing_jobs:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "El documento tiene trabajos en ejecución. "
                    "No puede eliminarse mientras esté siendo procesado."
                ),
            )

        logger.info(
            f"[RAG] Jobs asociados: {len(jobs)}"
        )

        # =====================================================
        # 4. Comprobar archivo físico
        # =====================================================

        if filepath.exists():
            if not filepath.is_file():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "La ruta registrada del documento "
                        "no corresponde a un archivo."
                    ),
                )
            logger.info(
                "[RAG] Archivo físico encontrado."
            )
        else:
            logger.info(
                f"[RAG] Advertencia: el archivo físico {filepath}"
                "no existe actualmente."
            )

        # =====================================================
        # 5. Determinar si es documento OCR
        # =====================================================
        #
        # No dependemos únicamente del nombre "_texto.pdf".
        #
        # La relación oficial está en:
        #
        # documents.ocr_filepath
        #
        # =====================================================

        conn = get_connection()

        original_row = conn.execute(
            """
            SELECT
                id,
                user_id,
                filename,
                filepath,
                ocr_filepath
            FROM documents
            WHERE user_id = ?
            AND ocr_filepath = ?
            LIMIT 1
            """,
            (
                user_id,
                str(filepath),
            ),
        ).fetchone()

        conn.close()
        conn = None

        is_ocr_document = (
            original_row is not None
        )

        original_document = (
            dict(original_row)
            if original_row
            else None
        )

        if is_ocr_document:

            logger.info(
                "[RAG] El documento es una versión OCR."
            )
            logger.info(
                f"[RAG] Documento original: "
                f"{original_document['id']}"
            )
            logger.info(
                f"[RAG] Archivo original: "
                f"{original_document['filename']}"
            )
        else:
            logger.info("")
            logger.info(
                "[RAG] El documento no es una versión OCR."
            )

        # =====================================================
        # 6. Comprobar LanceDB
        # =====================================================

        logger.info(
            "[RAG] Comprobando LanceDB..."
        )

        lancedb_exists = False
        db = lancedb.connect(
            str(LANCEDB_DIR)
        )

        tables = db.list_tables()

        if "chunks" in tables.tables:

            lancedb_exists = True

            logger.info(
                "[RAG] Tabla LanceDB encontrada."
            )

        else:

            logger.warning(
                "[RAG] La tabla LanceDB no existe."
            )

        # =====================================================
        # 7. FASE DE COMPROBACIÓN TERMINADA
        # =====================================================
        logger.info("")
        logger.info(
            "[RAG] Todas las comprobaciones terminaron."
        )
        logger.info(
            "[RAG] Iniciando eliminación..."
        )

        # =====================================================
        # =====================================================
        # FASE 2
        # ELIMINACIÓN
        # =====================================================
        # =====================================================

        # =====================================================
        # 8. Eliminar archivo físico
        # =====================================================

        physical_file_deleted = False

        if filepath.exists():
            logger.info("")
            logger.info(
                "[RAG] Eliminando archivo físico..."
            )
            filepath.unlink()
            physical_file_deleted = True
            logger.info(
                f"[RAG] Archivo eliminado: {filepath}"
            )
        else:
            logger.error(
                f"[RAG] No había archivo físico que eliminar. {filepath}"
            )

        # =====================================================
        # 9. Eliminar embeddings de LanceDB
        # =====================================================

        embeddings_deleted = False

        if lancedb_exists:
            logger.info("")
            logger.info(
                "[RAG] Eliminando embeddings..."
            )
            table = db.open_table(
                "chunks"
            )

            # document_id proviene de SQLite y es un UUID.
            # No contiene comillas simples.
            table.delete(
                f"document_id = '{document_id}'"
            )
            embeddings_deleted = True
            logger.info(
                "[RAG] Embeddings eliminados."
            )

        # =====================================================
        # 10. Eliminar FTS5
        # =====================================================

        logger.info("")
        logger.info(
            "[RAG] Eliminando registros FTS5..."
        )

        conn = get_connection()

        try:
            cursor = conn.execute(
                """
                DELETE FROM chunks_fts
                WHERE document_id = ?
                """,
                (
                    document_id,
                ),
            )

            fts_deleted_count = cursor.rowcount
            conn.commit()

        finally:
            conn.close()
            conn = None

        logger.info(
            f"[RAG] Registros FTS5 eliminados: "
            f"{fts_deleted_count}"
        )

        # =====================================================
        # 11. Si es documento OCR:
        #
        # quitar la relación del documento original.
        #
        # NO eliminar el documento original.
        # =====================================================

        original_reference_cleared = False

        if is_ocr_document:

            logger.info("")
            logger.info(
                "[RAG] Eliminando referencia OCR "
                "del documento original..."
            )
            conn = get_connection()

            try:
                cursor = conn.execute(
                    """
                    UPDATE documents
                    SET ocr_filepath = NULL
                    WHERE id = ?
                    AND user_id = ?
                    AND ocr_filepath = ?
                    """,
                    (
                        original_document["id"],
                        user_id,
                        str(filepath),
                    ),
                )
                conn.commit()

                original_reference_cleared = (
                    cursor.rowcount == 1
                )

            finally:

                conn.close()
                conn = None

            logger.info(
                "[RAG] ocr_filepath del documento original "
                "establecido en NULL."
            )

        # =====================================================
        # 12. Eliminar jobs
        # =====================================================
        logger.info("")
        logger.info(
            "[RAG] Eliminando jobs asociados..."
        )

        conn = get_connection()

        try:
            cursor = conn.execute(
                """
                DELETE FROM jobs
                WHERE document_id = ?
                """,
                (
                    document_id,
                ),
            )

            jobs_deleted_count = cursor.rowcount

            conn.commit()

        finally:
            conn.close()
            conn = None
        logger.info(
            f"[RAG] Jobs eliminados: "
            f"{jobs_deleted_count}"
        )

        # =====================================================
        # 13. Eliminar documento de SQLite
        # =====================================================

        logger.info("")
        logger.info(
            "[RAG] Eliminando documento de SQLite..."
        )

        conn = get_connection()

        try:
            cursor = conn.execute(
                """
                DELETE FROM documents
                WHERE id = ?
                AND user_id = ?
                """,
                (
                    document_id,
                    user_id,
                ),
            )
            document_deleted = (
                cursor.rowcount == 1
            )
            conn.commit()

        finally:
            conn.close()
            conn = None

        if not document_deleted:
            logger.error(
                        "[RAG] Registro documents eliminado."
                    )
            raise RuntimeError(
                "El archivo y los índices fueron eliminados, "
                "pero no se pudo eliminar el registro "
                "documents."
            )
        logger.info(
            "[RAG] Registro documents eliminado."
        )

        # =====================================================
        # 14. Resultado
        # =====================================================

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "[RAG] DOCUMENTO ELIMINADO CORRECTAMENTE"
        )
        logger.info("=" * 60)

        return {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "deleted": True,
            "physical_file_deleted": (
                physical_file_deleted
            ),
            "embeddings_deleted": (
                embeddings_deleted
            ),
            "fts_deleted": True,
            "jobs_deleted": (
                jobs_deleted_count
            ),
            "was_ocr_document": (
                is_ocr_document
            ),
            "original_document_id": (
                original_document["id"]
                if original_document
                else None
            ),
            "original_ocr_filepath_cleared": (
                original_reference_cleared
            ),
            "message": (
                f"Documento '{filename}' "
                "eliminado correctamente."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        logger.error(
            "Error eliminando documento %s: %s",
            document_id,
            str(e),
        )
        logger.error(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Error al eliminar el documento: "
                f"{str(e)}"
            ),
        )


# ============================================================
# categorias
# ============================================================


@app.get("/categorias")
async def get_categories(
    api_key: str = Depends(get_api_key),
    x_client_id: str = Header(
        ...,
        alias=CLIENT_ID_HEADER_NAME,
        description="ID único del cliente/usuario",
    ),
):
    try:

        # =====================================================
        # IDENTIDAD DEL USUARIO
        # =====================================================

        user_id = "".join(
            c
            for c in x_client_id
            if c.isalnum() or c in ("_", "-")
        )

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Identificador de cliente "
                    "(X-Client-ID) inválido."
                ),
            )

        # =====================================================
        # ASEGURAR QUE EXISTE "GENERAL"
        # =====================================================
        #
        # Cada usuario debe tener siempre su propia
        # categoría "general".
        #
        # get_or_create_general_category() la crea
        # automáticamente si todavía no existe.
        # =====================================================

        get_or_create_general_category(
            user_id=user_id
        )

        # =====================================================
        # OBTENER CATEGORÍAS DEL USUARIO
        # =====================================================

        categories = list_categories(
            user_id=user_id
        )

        logger.info(
            "[CATEGORIA] Listando categorías - "
            "user_id=%s, count=%d",
            user_id,
            len(categories),
        )

        # =====================================================
        # LOG DE CATEGORÍAS ENCONTRADAS
        # =====================================================

        logger.info("=" * 60)
        logger.info("[CATEGORIA] Categorías encontradas")
        logger.info("=" * 60)
        logger.info(
            "[CATEGORIA] Usuario: %s",
            user_id,
        )
        logger.info(
            "[CATEGORIA] Total: %d",
            len(categories),
        )

        for index, category in enumerate(
            categories,
            start=1,
        ):
            logger.info(
                "[CATEGORIA] #%d | id=%s | nombre=%s | documentos=%s",
                index,
                category.get("id"),
                category.get("name"),
                category.get("document_count", 0),
            )

        logger.info("=" * 60)

        # =====================================================
        # RESPUESTA
        # =====================================================

        return {
            "success": True,
            "user_id": user_id,
            "categorias": categories,
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "Error listando categorías "
            "para usuario %s",
            x_client_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Error listando categorías: {str(e)}"
            ),
        )

class CategoryRequest(BaseModel):
    nombre: str

@app.post("/categorias")
async def add_category(
    request: CategoryRequest,
    api_key: str = Depends(get_api_key),
    x_client_id: str = Header(
        ...,
        alias=CLIENT_ID_HEADER_NAME,
        description="ID único del cliente/usuario",
    ),
):
    try:

        # =====================================================
        # VALIDAR / LIMPIAR USER ID
        # =====================================================

        user_id = "".join(
            c
            for c in x_client_id
            if c.isalnum() or c in ("_", "-")
        )

        if not user_id:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Identificador de cliente inválido.",
            )

        # =====================================================
        # CREAR CATEGORÍA PARA ESTE USUARIO
        # =====================================================

        category_id, category_name = create_category(
            user_id=user_id,
            name=request.nombre,
        )

        logger.info(
            "[CATEGORIA] Creada - "
            "user_id=%s | id=%s | nombre=%s",
            user_id,
            category_id,
            category_name,
        )

        return {
            "success": True,
            "categoria": {
                "id": category_id,
                "nombre": category_name,
            },
        }

    except HTTPException:
        raise

    except ValueError as e:

        # Por ejemplo:
        # - nombre vacío
        # - categoría duplicada para este usuario

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    except Exception as e:

        logger.exception(
            "Error creando categoría"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando categoría: {str(e)}",
        )


@app.delete("/categorias/{categoria_id}")
async def remove_category(
    categoria_id: int,
    api_key: str = Depends(get_api_key),
    x_client_id: str = Header(
        ...,
        alias=CLIENT_ID_HEADER_NAME,
        description="ID único del cliente/usuario",
    ),
):
    try:

        # =====================================================
        # VALIDAR / LIMPIAR USER ID
        # =====================================================

        user_id = "".join(
            c
            for c in x_client_id
            if c.isalnum() or c in ("_", "-")
        )

        if not user_id:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Identificador de cliente inválido.",
            )

        # =====================================================
        # ELIMINAR CATEGORÍA DEL USUARIO
        # =====================================================

        delete_category(
            user_id=user_id,
            category_id=categoria_id,
        )

        logger.info(
            "[CATEGORIA] Eliminada - "
            "user_id=%s | category_id=%s",
            user_id,
            categoria_id,
        )

        return {
            "success": True,
            "categoria_id": categoria_id,
            "message": (
                "Categoría eliminada correctamente. "
                "Los documentos asociados fueron "
                "movidos a 'general'."
            ),
        }

    except HTTPException:
        raise

    except ValueError as e:

        # Puede ocurrir si:
        # - la categoría no existe para este usuario
        # - se intenta eliminar 'general'

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:

        logger.exception(
            "Error eliminando categoría"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error eliminando categoría: {str(e)}",
        )



class UpdateDocumentCategoryRequest(BaseModel):
    hash: str
    category_id: int


@app.put("/document/category")
async def update_document_category_endpoint(
    request: UpdateDocumentCategoryRequest,
    api_key: str = Depends(get_api_key),
    user_dir: str = Depends(get_client_directory),
):
    """
    Actualiza la categoría de un documento utilizando su hash.

    El servidor utiliza:

        X-Client-ID + hash
                    ↓
             document_id
                    ↓
        update_document_category()
    """

    try:

        # =====================================================
        # IDENTIDAD DEL USUARIO
        # =====================================================

        user_id = os.path.basename(
            os.path.normpath(user_dir)
        )

        # =====================================================
        # VALIDAR HASH
        # =====================================================

        document_hash = request.hash.strip()

        if not document_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El hash del documento no puede estar vacío.",
            )

        # =====================================================
        # VALIDAR CATEGORY ID
        # =====================================================

        if request.category_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El category_id debe ser mayor que cero.",
            )

        logger.info(
            "[CATEGORIA] Solicitud de actualización - "
            "user_id=%s, hash=%s, category_id=%s",
            user_id,
            document_hash,
            request.category_id,
        )

        # =====================================================
        # BUSCAR DOCUMENTO POR USER_ID + HASH
        # =====================================================

        conn = get_connection()

        try:
            document_row = conn.execute(
                """
                SELECT
                    id,
                    filename,
                    hash,
                    category_id
                FROM documents
                WHERE user_id = ?
                AND hash = ?
                LIMIT 1
                """,
                (
                    user_id,
                    document_hash,
                ),
            ).fetchone()

        finally:
            conn.close()

        # =====================================================
        # DOCUMENTO NO ENCONTRADO
        # =====================================================

        if document_row is None:

            logger.warning(
                "[CATEGORIA] Documento no encontrado - "
                "user_id=%s, hash=%s",
                user_id,
                document_hash,
            )

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "No existe ningún documento con el hash "
                    "proporcionado para este cliente."
                ),
            )

        document = dict(document_row)

        # El document_id solamente existe internamente
        # en el servidor.
        document_id = document["id"]

        logger.info(
            "[CATEGORIA] Documento encontrado - "
            "document_id=%s, filename=%s, categoría_actual=%s",
            document_id,
            document["filename"],
            document["category_id"],
        )

        # =====================================================
        # ACTUALIZAR CATEGORÍA
        # =====================================================

        updated = update_document_category(
            user_id=user_id,
            document_id=document_id,
            category_id=request.category_id,
        )

        # =====================================================
        # COMPROBAR ACTUALIZACIÓN
        # =====================================================

        if not updated:

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No se pudo actualizar la categoría "
                    f"del documento '{document['filename']}'."
                ),
            )

        # =====================================================
        # LOG
        # =====================================================

        logger.info(
            "[CATEGORIA] Categoría actualizada correctamente - "
            "user_id=%s, filename=%s, hash=%s, "
            "category_id=%s",
            user_id,
            document["filename"],
            document_hash,
            request.category_id,
        )

        # =====================================================
        # RESPUESTA
        # =====================================================

        return {
            "success": True,
            "hash": document_hash,
            "filename": document["filename"],
            "category_id": request.category_id,
            "message": (
                "Categoría del documento "
                "actualizada correctamente."
            ),
        }

    except HTTPException:
        raise

    except ValueError as e:

        logger.warning(
            "[CATEGORIA] Categoría inválida - "
            "user_id=%s, hash=%s, category_id=%s: %s",
            user_id,
            document_hash,
            request.category_id,
            e,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:

        logger.exception(
            "[CATEGORIA] Error actualizando categoría"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Error actualizando categoría: {str(e)}"
            ),
        )




def datetime_to_str(mtime: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')




# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    import uvicorn
    # Obtener IP local de la LAN para escuchar peticiones en la red
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()

    

    # Rutas para los certificados SSL (busca en el ejecutable o carpeta local)

    cert_file = Path(__file__).resolve().parent / "cert.pem"
    key_file = Path(__file__).resolve().parent / "key.pem"

    logger.info("\n" + "="*60)
    logger.info(f"Iniciando Servidor en: https://{IP}:8005")
    logger.info(f"API Key configurada: {API_KEY}")
    logger.info(f"Directorio de Almacenamiento: {BASE_UPLOAD_DIR}")
    logger.info("="*60 + "\n")

    # Ejecutar Uvicorn con configuración SSL
    uvicorn.run(
        app,
        host="0.0.0.0",  # Escuchar en todas las interfaces de red de la máquina
        port=8005,
        ssl_keyfile=key_file,
        ssl_certfile=cert_file,
        timeout_keep_alive=60,
        limit_concurrency=100,
        reload=False
    )
