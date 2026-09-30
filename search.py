from .config import LANCEDB_DIR
from .database import exact_search, identifier_search,get_document_hashes,search_documents_by_filename
from .embeddings import encode_query
import lancedb
import re


# ============================================================
# IDENTIFICADORES
# ============================================================

def extract_identifiers(query: str) -> list[str]:
    """
    Extrae identificadores de una consulta.

    Detecta:

    - Números largos
      1324808388

    - Tokens alfanuméricos
      FACTURA2026
      ABC123456

    - Identificadores con separadores
      ABC-2026-001
      MX/ABC/123456
      MX/ABC-123/2026
      UUID

    - Emails
      usuario@empresa.com

    - Hashtags
      #FACTURA2026

    - Teléfonos
      +52 33 1234 5678
      +52-33-1234-5678

    La consulta original NO se modifica.
    """

    if not query:
        return []

    query = query.strip()

    identifiers = []

    # ============================================================
    # 1. EMAILS
    # ============================================================

    emails = re.findall(
        r'(?<![\w.+-])'
        r'[A-Za-z0-9._%+-]+'
        r'@'
        r'[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
        r'(?![\w.-])',
        query,
    )

    identifiers.extend(emails)

    # ============================================================
    # 2. HASHTAGS
    # ============================================================

    hashtags = re.findall(
        r'(?<!\w)'
        r'#[A-Za-z0-9][A-Za-z0-9_-]{3,}',
        query,
    )

    identifiers.extend(hashtags)

    # ============================================================
    # 3. TELÉFONOS
    # ============================================================

    phones = re.findall(
        r'(?<!\w)'
        r'\+\d{1,3}'
        r'(?:[\s.-]?\(?\d{1,4}\)?){2,6}'
        r'(?!\w)',
        query,
    )

    identifiers.extend(phones)

    # ============================================================
    # 4. IDENTIFICADORES CON SEPARADORES
    #
    # Ejemplos:
    #
    # ABC-2026-001
    # MX/ABC/123456
    # MX/ABC-123/2026
    # FACTURA_2026_001
    # UUID
    # ============================================================

    separated_tokens = re.findall(
        r'(?<![\w])'
        r'[A-Za-z0-9]+'
        r'(?:[/_.:-][A-Za-z0-9]+)+'
        r'(?![\w])',
        query,
    )

    for token in separated_tokens:

        compact = re.sub(
            r'[^A-Za-z0-9]',
            '',
            token,
        )

        # Demasiado pequeño para ser un identificador
        if len(compact) < 6:
            continue

        has_digit = any(
            c.isdigit()
            for c in compact
        )

        has_alpha = any(
            c.isalpha()
            for c in compact
        )

        # Debe contener números
        if not has_digit:
            continue

        # Si solamente son números, exigimos mínimo 6
        if not has_alpha and len(compact) < 6:
            continue

        identifiers.append(token)

    # ============================================================
    # 5. TOKENS ALFANUMÉRICOS LARGOS
    #
    # ABC123456
    # FACTURA2026
    # 123ABC456
    # ============================================================

    alphanumeric_tokens = re.findall(
        r'(?<![\w])'
        r'[A-Za-z0-9]{6,}'
        r'(?![\w])',
        query,
    )

    for token in alphanumeric_tokens:

        has_digit = any(
            c.isdigit()
            for c in token
        )

        has_alpha = any(
            c.isalpha()
            for c in token
        )

        if has_digit:
            identifiers.append(token)

    # ============================================================
    # 6. NÚMEROS LARGOS
    #
    # 1324808388
    # 444050804041
    # ============================================================

    numeric_tokens = re.findall(
        r'(?<![\w])'
        r'\d{6,}'
        r'(?![\w])',
        query,
    )

    identifiers.extend(numeric_tokens)

    # ============================================================
    # 7. DEDUPLICAR
    # ============================================================

    unique = []
    seen = set()

    for identifier in identifiers:

        identifier = identifier.strip()

        if not identifier:
            continue

        key = identifier.lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(identifier)

    return unique

def debug_identifiers(query: str):
    identifiers = extract_identifiers(query)

    print()
    print("=" * 70)
    print("DEBUG IDENTIFICADORES")
    print("QUERY:", repr(query))
    print("IDENTIFICADORES:", identifiers)
    print("=" * 70)
    print()

    return identifiers


def is_identifier_query(query: str) -> bool:
    """
    Indica si la consulta contiene al menos un identificador.

    Es solamente una señal auxiliar.

    No cambia el modo de búsqueda.
    """

    return bool(
        extract_identifiers(query)
    )


# ============================================================
# LANCEDB
# ============================================================

def get_vector_table():
    """Abre la tabla de vectores de LanceDB."""

    db = lancedb.connect(
        str(LANCEDB_DIR)
    )

    tables = db.list_tables()

    if "chunks" not in tables.tables:
        return None

    return db.open_table("chunks")


def result_id(result):
    """
    Genera un identificador estable para un chunk.
    """

    return (
        f"{result['document_id']}:"
        f"{result['page']}:"
        f"{result['line_start']}:"
        f"{result['line_end']}"
    )


def normalize_semantic_result(result):
    """
    Convierte un resultado de LanceDB al formato interno.
    """

    result = dict(result)

    distance = result.get("distance")

    if distance is None:
        distance = result.get("_distance")

    return {
        "id": result_id(result),
        "document_id": result.get("document_id"),
        "user_id": result.get("user_id"),
        "filename": result.get("filename"),
        "page": result.get("page"),
        "line_start": result.get("line_start"),
        "line_end": result.get("line_end"),
        "text": result.get("text"),
        "distance": distance,

        "semantic_score": None,

        "exact_score": 0.0,

        # Señales adicionales.
        "exact_query_score": 0.0,
        "identifier_score": 0.0,

        "match_type": "semantic",
    }


def attach_document_hashes(results):
    """
    Agrega el hash almacenado en documentsBD a los resultados.

    Hace UNA sola consulta SQLite para todos los document_id
    encontrados.
    
    """

    if not results:
        return results

    document_ids = []

    for result in results:

        document_id = result.get("document_id")

        if document_id:
            document_ids.append(document_id)

    if not document_ids:
        return results

    hashes = get_document_hashes(
        document_ids
    )

    for result in results:

        document_id = result.get(
            "document_id"
        )

        result["hash"] = hashes.get(
            document_id
        )

    return results


def limit_results_per_document(
    results: list,
    max_per_document: int = 2,
) -> list:
    """
    Limita la cantidad de resultados por documento.

    Máximo:
        2 resultados del mismo document_id.

    Mantiene el orden original de los resultados.
    Por lo tanto, si los resultados ya están ordenados
    por score, se conservan los mejores resultados
    de cada documento.

    La cantidad total final puede ser menor que 'limit'.
    """

    if not results:
        return []

    document_counts = {}
    filtered_results = []

    for result in results:

        document_id = result.get("document_id")

        # Si no existe document_id, no podemos agruparlo.
        # Lo dejamos pasar.
        if not document_id:

            filtered_results.append(result)

            continue

        current_count = document_counts.get(
            document_id,
            0,
        )

        if current_count >= max_per_document:

            continue

        filtered_results.append(result)

        document_counts[document_id] = (
            current_count + 1
        )

    return filtered_results


def semantic_search(
    user_id: str,
    query: str,
    limit: int = 10,
):
    """
    Búsqueda semántica mediante LanceDB.

    IMPORTANTE:
    La consulta utilizada para generar el embedding es
    EXACTAMENTE la consulta original.

    La detección de identificadores NO modifica esta búsqueda.
    """
    print("search.py -> semantic_search")
    table = get_vector_table()

    if table is None:
        return []

    # ========================================================
    # CONSULTA SEMÁNTICA ORIGINAL
    # ========================================================

    vector = encode_query(query)

    candidates = (
        table
        .search(
            vector.tolist(),
            vector_column_name="vector",
        )
        .limit(limit * 5)
        .to_list()
    )

    # ---------------------------------------------------------
    # Filtrar por usuario
    # ---------------------------------------------------------

    candidates = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("user_id") == user_id
    ]

    if not candidates:
        return []

    # ---------------------------------------------------------
    # Distancias
    # ---------------------------------------------------------

    distances = []

    for candidate in candidates:

        distance = candidate.get("_distance")

        if distance is None:
            distance = candidate.get("distance")

        if distance is not None:
            distances.append(
                float(distance)
            )

    if not distances:
        return []

    min_distance = min(distances)
    max_distance = max(distances)

    # ---------------------------------------------------------
    # Normalización
    # ---------------------------------------------------------

    normalized_candidates = []

    for candidate in candidates:

        distance = candidate.get("_distance")

        if distance is None:
            distance = candidate.get("distance")

        if distance is None:

            semantic_score = 0.0

        elif max_distance == min_distance:

            semantic_score = 1.0

        else:

            semantic_score = (
                (max_distance - float(distance))
                /
                (max_distance - min_distance)
            )

        normalized = normalize_semantic_result(
            candidate
        )

        normalized["semantic_score"] = (
            semantic_score
        )

        normalized_candidates.append(
            normalized
        )

    # ---------------------------------------------------------
    # Ordenar
    # ---------------------------------------------------------

    normalized_candidates.sort(
        key=lambda result:
            result["semantic_score"],
        reverse=True,
    )


    # print()
    # print("=" * 100)
    # print("[RAG][SEMANTIC] CANDIDATOS NORMALIZADOS ORDENADOS")
    # print("=" * 100)

    # for i, result in enumerate(
    #     normalized_candidates,
    #     start=1,
    # ):
    #     if(i>20): break  #limitar a 20 prints

    #     print(
    #         f"[RAG][SEMANTIC] #{i:03d} "
    #         f"semantic_score={result.get('semantic_score', 0):.6f} | "
    #         f"distance={result.get('distance')} | "
    #         f"document_id={result.get('document_id')} | "
    #         f"page={result.get('page')} | "
    #         f"line={result.get('line_start')}-{result.get('line_end')} | "
    #         f"filename={result.get('filename')}"
    #     )

    # print("=" * 100)
    # print(
    #     f"[RAG][SEMANTIC] Total candidatos: "
    #     f"{len(normalized_candidates)}"
    # )
    # print("=" * 100)
    # print()


    return normalized_candidates[:limit]


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    user_id: str,
    query: str,
    limit: int = 10,
    exact_percentage: int = 30,
    semantic_percentage: int = 70,
    exact_min_score: float = 0.60,
    semantic_min_score: float = 0.60,
):
    """
    Búsqueda híbrida.

    La cantidad de resultados se divide entre:

        exact_percentage
        semantic_percentage

    Los límites representan DOCUMENTOS ÚNICOS.

    Ejemplo:

        limit = 10
        exact_percentage = 30
        semantic_percentage = 70

        exact_limit    = 3
        semantic_limit = 7

    Cada lista puede contener solamente una aparición de cada
    document_id.

    Además:

        semantic_min_score
            Score mínimo permitido para resultados semánticos.

        exact_min_score
            Score mínimo permitido para resultados exactos.

    IMPORTANTE:

    El mismo documento puede aparecer una vez en EXACTOS y una
    vez en SEMÁNTICOS.

    Ejemplo válido:

        EXACT:
            D1
            D2
            D3

        SEMANTIC:
            D1
            D4
            D5

    Resultado:

        D1
        D2
        D3
        D1
        D4
        D5
    """

    print("search.py -> hybrid_search")

    debug_identifiers(query)

    # ========================================================
    # VALIDAR PORCENTAJES
    # ========================================================

    if exact_percentage < 0:
        raise ValueError(
            "exact_percentage no puede ser negativo"
        )

    if semantic_percentage < 0:
        raise ValueError(
            "semantic_percentage no puede ser negativo"
        )

    if (
        exact_percentage
        +
        semantic_percentage
        != 100
    ):
        raise ValueError(
            "exact_percentage + semantic_percentage "
            "debe ser exactamente 100"
        )

    # ========================================================
    # CALCULAR LÍMITES
    # ========================================================

    exact_limit = int(
        limit * exact_percentage / 100
    )

    semantic_limit = (
        limit
        -
        exact_limit
    )

    print()
    print( "[RAG][HYBRID] Distribución de resultados"    )

    print(        f"[RAG][HYBRID] limit = {limit}"    )

    print(        f"[RAG][HYBRID] exact_percentage = "        f"{exact_percentage}%"    )

    print(
        f"[RAG][HYBRID] semantic_percentage = "
        f"{semantic_percentage}%"
    )

    print(
        f"[RAG][HYBRID] exact_limit = "
        f"{exact_limit}"
    )

    print(
        f"[RAG][HYBRID] semantic_limit = "
        f"{semantic_limit}"
    )

    print(
        f"[RAG][HYBRID] exact_min_score = "
        f"{exact_min_score}"
    )

    print(
        f"[RAG][HYBRID] semantic_min_score = "
        f"{semantic_min_score}"
    )

    # ========================================================
    # 1. BÚSQUEDA SEMÁNTICA
    # ========================================================
    #
    # Pedimos una cantidad grande de candidatos.
    #
    # No utilizamos semantic_limit aquí porque semantic_limit
    # representa DOCUMENTOS ÚNICOS y LanceDB devuelve CHUNKS.
    #
    # Necesitamos suficientes candidatos para poder eliminar
    # duplicados por document_id.
    # ========================================================

    semantic_candidate_limit = max(
        limit * 10,
        50,
    )

    semantic_results = semantic_search(
        user_id=user_id,
        query=query,
        limit=semantic_candidate_limit,
    )

    # ========================================================
    # 2. OBTENER DOCUMENTOS SEMÁNTICOS ÚNICOS
    # ========================================================

    semantic_unique_results = []

    semantic_document_ids = set()

    for result in semantic_results:

        semantic_score = float(
            result.get(
                "semantic_score",
                0.0,
            )
        )

        # ----------------------------------------------------
        # SCORE MÍNIMO
        # ----------------------------------------------------

        if semantic_score < semantic_min_score:

            # print(
            #     "[RAG][HYBRID] SEMANTIC DESCARTADO "
            #     f"por score: "
            #     f"{semantic_score:.6f} "
            #     f"< "
            #     f"{semantic_min_score:.6f} "
            #     f"| document_id="
            #     f"{result.get('document_id')}"
            # )

            continue

        document_id = result.get(
            "document_id"
        )

        # ----------------------------------------------------
        # DOCUMENT_ID OBLIGATORIO
        # ----------------------------------------------------

        if document_id is None:

            print(
                "[RAG][HYBRID] SEMANTIC DESCARTADO "
                "porque no tiene document_id"
            )

            continue

        # ----------------------------------------------------
        # ELIMINAR DUPLICADOS
        # ----------------------------------------------------

        if document_id in semantic_document_ids:

            continue

        # ----------------------------------------------------
        # NUEVO DOCUMENTO
        # ----------------------------------------------------

        semantic_document_ids.add(
            document_id
        )

        result = dict(
            result
        )

        result["match_type"] = "semantic"

        result["exact_score"] = 0.0

        result["exact_query_score"] = 0.0

        result["identifier_score"] = 0.0

        semantic_unique_results.append(
            result
        )

        # ----------------------------------------------------
        # YA TENEMOS SUFICIENTES DOCUMENTOS
        # ----------------------------------------------------

        if (
            len(
                semantic_unique_results
            )
            >= semantic_limit
        ):

            break

    # ========================================================
    # DEBUG SEMÁNTICOS
    # ========================================================

    print()
    print(
        "[RAG][HYBRID] RESULTADOS SEMÁNTICOS"
    )

    for index, result in enumerate(
        semantic_unique_results,
        start=1,
    ):

        print(
            "[RAG][HYBRID][SEMANTIC] "
            f"#{index:03d} "
            f"score="
            f"{float(result.get('semantic_score', 0.0)):.6f} "
            f"document_id="
            f"{result.get('document_id')} "
            f"filename="
            f"{result.get('filename')} "
            f"page="
            f"{result.get('page')}"
        )

    # ========================================================
    # 3. CONSULTAS EXACTAS
    # ========================================================

    exact_queries = [
        {
            "query": query,
            "source": "query",
        }
    ]

    identifiers = extract_identifiers(
        query
    )

    for identifier in identifiers:

        if (
            identifier.strip().lower()
            ==
            query.lower()
        ):
            continue

        exact_queries.append(
            {
                "query": identifier,
                "source": "identifier",
            }
        )

    # ========================================================
    # 4. OBTENER RESULTADOS EXACTOS
    # ========================================================
    #
    # Los resultados exactos pueden proceder de:
    #
    #   1. Contenido textual     -> exact_search()
    #   2. Nombre del documento  -> search_documents_by_filename()
    #   3. Identificadores       -> identifier_search()
    #
    # Todos participan en la misma cuota EXACT.
    #
    # Los resultados se deduplican por document_id.
    #
    # Se siguen recorriendo resultados hasta obtener
    # exact_limit DOCUMENTOS ÚNICOS que superen
    # exact_min_score.
    # ========================================================

    exact_unique_results = []

    exact_document_ids = set()

    for search_item in exact_queries:

        exact_query = search_item[
            "query"
        ]

        source = search_item[
            "source"
        ]

        # ----------------------------------------------------
        # SOLAMENTE CONTINUAMOS SI TODAVÍA NECESITAMOS
        # DOCUMENTOS EXACTOS
        # ----------------------------------------------------

        if (
            len(
                exact_unique_results
            )
            >= exact_limit
        ):
            break

        # ----------------------------------------------------
        # OBTENER MATCHES
        # ----------------------------------------------------
        #
        # query:
        #   - busca en contenido
        #   - busca en filename
        #
        # identifier:
        #   - busca solamente identificadores
        # ----------------------------------------------------

        if source == "query":

            candidate_limit = max(
                exact_limit * 10,
                50,
            )

            # -----------------------------------------------
            # BUSCAR EN CONTENIDO
            # -----------------------------------------------

            content_matches = exact_search(
                user_id=user_id,
                query=exact_query,
                limit=candidate_limit,
            )

            # -----------------------------------------------
            # BUSCAR EN NOMBRE DE ARCHIVO
            # -----------------------------------------------

            filename_matches = (
                search_documents_by_filename(
                    user_id=user_id,
                    query=exact_query,
                    limit=candidate_limit,
                )
            )

            # ------------------------------------------------
            # COMBINAR AMBAS FUENTES
            # ------------------------------------------------
            #
            # Primero contenido y después filename.
            #
            # Si un documento aparece en ambas fuentes,
            # la deduplicación posterior conservará el
            # primer resultado.
            # ------------------------------------------------

            matches = (
                [
                    (
                        result,
                        "query",
                    )
                    for result in content_matches
                ]
                +
                [
                    (
                        result,
                        "filename",
                    )
                    for result in filename_matches
                ]
            )

        else:

            # ------------------------------------------------
            # BÚSQUEDA POR IDENTIFICADOR
            # ------------------------------------------------

            identifier_matches = identifier_search(
                user_id=user_id,
                identifier=exact_query,
                limit=max(
                    exact_limit * 10,
                    50,
                ),
            )

            matches = [
                (
                    result,
                    "identifier",
                )
                for result in identifier_matches
            ]

        # ----------------------------------------------------
        # PROCESAR MATCHES
        # ----------------------------------------------------

        for result, result_source in matches:

            exact_score = float(
                result.get(
                    "exact_score",
                    1.0,
                )
            )

            # ------------------------------------------------
            # SCORE MÍNIMO
            # ------------------------------------------------

            if (
                exact_score
                <
                exact_min_score
            ):

                print(
                    "[RAG][HYBRID] EXACT DESCARTADO "
                    f"por score: "
                    f"{exact_score:.6f} "
                    f"< "
                    f"{exact_min_score:.6f} "
                    f"| document_id="
                    f"{result.get('document_id')}"
                )

                continue

            document_id = result.get(
                "document_id"
            )

            if document_id is None:

                print(
                    "[RAG][HYBRID] EXACT DESCARTADO "
                    "porque no tiene document_id"
                )

                continue

            # ------------------------------------------------
            # ELIMINAR DUPLICADOS
            # ------------------------------------------------
            #
            # La deduplicación es por document_id.
            #
            # Un documento puede haber coincidido:
            #
            #   - en contenido
            #   - en filename
            #   - en identifier
            #
            # pero solamente aparecerá una vez dentro
            # de la lista EXACT.
            # ------------------------------------------------

            if document_id in exact_document_ids:

                continue

            # ------------------------------------------------
            # NORMALIZAR RESULTADO
            # ------------------------------------------------

            result = dict(
                result
            )

            result["semantic_score"] = 0.0

            result["distance"] = None

            result["exact_score"] = exact_score

            result["exact_query_score"] = 0.0

            result["identifier_score"] = 0.0

            # ------------------------------------------------
            # TIPO DE MATCH
            # ------------------------------------------------

            if result_source == "query":

                result[
                    "exact_query_score"
                ] = exact_score

                result[
                    "match_type"
                ] = "exact"

            elif result_source == "filename":

                result[
                    "exact_query_score"
                ] = exact_score

                result[
                    "match_type"
                ] = "filename"

            elif result_source == "identifier":

                result[
                    "identifier_score"
                ] = exact_score

                result[
                    "match_type"
                ] = "exact"

            # ------------------------------------------------
            # REGISTRAR DOCUMENTO
            # ------------------------------------------------

            exact_document_ids.add(
                document_id
            )

            exact_unique_results.append(
                result
            )

            # ------------------------------------------------
            # YA TENEMOS SUFICIENTES
            # ------------------------------------------------

            if (
                len(
                    exact_unique_results
                )
                >= exact_limit
            ):

                break

    # ========================================================
    # DEBUG EXACTOS
    # ========================================================

    print()
    print(
        "[RAG][HYBRID] RESULTADOS EXACTOS"
    )

    for index, result in enumerate(
        exact_unique_results,
        start=1,
    ):

        print(
            "[RAG][HYBRID][EXACT] "
            f"#{index:03d} "
            f"score="
            f"{float(result.get('exact_score', 0.0)):.6f} "
            f"document_id="
            f"{result.get('document_id')} "
            f"filename="
            f"{result.get('filename')} "
            f"page="
            f"{result.get('page')}"
        )

    # ========================================================
    # 5. RESULTADO FINAL
    # ========================================================
    #
    # IMPORTANTE:
    #
    # NO combinamos scores.
    #
    # NO volvemos a ordenar.
    #
    # NO permitimos que un resultado semántico desplace
    # a uno exacto.
    #
    # Simplemente:
    #
    #     EXACTOS
    #     +
    #     SEMÁNTICOS
    #
    # respetando el orden obtenido por cada búsqueda.
    # ========================================================

    final_results = (
        exact_unique_results
        +
        semantic_unique_results
    )

    # ========================================================
    # DEBUG FINAL
    # ========================================================

    print()
    print(
        "[RAG][HYBRID] RESULTADOS FINALES"
    )

    print(
        f"[RAG][HYBRID] exactos = "
        f"{len(exact_unique_results)}"
    )

    print(
        f"[RAG][HYBRID] semánticos = "
        f"{len(semantic_unique_results)}"
    )

    print(
        f"[RAG][HYBRID] total = "
        f"{len(final_results)}"
    )

    for index, result in enumerate(
        final_results,
        start=1,
    ):

        print(
            "[RAG][HYBRID][FINAL] "
            f"#{index:03d} "
            f"type="
            f"{result.get('match_type')} "
            f"document_id="
            f"{result.get('document_id')} "
            f"filename="
            f"{result.get('filename')} "
            f"page="
            f"{result.get('page')} "
            f"exact="
            f"{float(result.get('exact_score', 0.0)):.6f} "
            f"semantic="
            f"{float(result.get('semantic_score', 0.0)):.6f}"
        )

    return final_results[:limit]


# ============================================================
# API GENERAL
# ============================================================

def search(
    user_id: str,
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
    exact_percentage: int = 30,
    semantic_percentage: int = 70,
    exact_min_score: float = 0.60,
    semantic_min_score: float = 0.60,
):
    """
    Funcion principal de busqueda
        exact_percentage  y  semantic_percentage
            definen que del total de resultados a entregar el N% sea de resultados exactos y 
            el T% sea de resultados semanticos

            por ejemplo si el sistema entrega 100 resultados 
                30 pueden ser exactos y 70 semanticos
                (obviamente si no se encuantran 30 exactos se daran menos o tal vez menos semanticos)

            esto es necesario porque los resultados semanticos siempre son muy abundantes y asi 
                mantenemos una proporcion mas saludable de resultados exactos y no saturar con semanticos        


    Parámetros híbridos:

        exact_percentage:
            Porcentaje de resultados reservado para exactos.

        semantic_percentage:
            Porcentaje de resultados reservado para semánticos.

        exact_min_score:
            Score mínimo permitido para resultados exactos.

        semantic_min_score:
            Score mínimo permitido para resultados semánticos.

    Los límites representan DOCUMENTOS ÚNICOS.

    Ejemplo:

        limit = 10
        exact_percentage = 30
        semantic_percentage = 70

        => máximo 3 documentos exactos
        => máximo 7 documentos semánticos

    El mismo document_id puede aparecer una sola vez en cada categoría.
    """

    print("search.py -> search")

    if not query or not query.strip():
        return []

    query = query.strip()

    # ========================================================
    # SEMANTIC
    # ========================================================

    if mode == "semantic":
        results = semantic_search(
            user_id=user_id,
            query=query,
            limit=limit,
        )

        # ----------------------------------------------------
        # FILTRAR SCORE MÍNIMO
        # ----------------------------------------------------

        results = [
            result
            for result in results
            if float(
                result.get(
                    "semantic_score",
                    0.0,
                )
            )
            >= semantic_min_score
        ]

        # ----------------------------------------------------
        # DOCUMENTOS ÚNICOS
        # ----------------------------------------------------

        unique_results = []
        seen_documents = set()

        for result in results:

            document_id = result.get(
                "document_id"
            )

            if (
                document_id is None
                or
                document_id in seen_documents
            ):
                continue

            seen_documents.add(
                document_id
            )

            unique_results.append(
                result
            )

            if (
                len(unique_results)
                >= limit
            ):
                break

        results = attach_document_hashes(
            unique_results
        )

        return results

    # ========================================================
    # EXACT
    # ========================================================

    if mode == "exact":

        results = exact_search(
            user_id=user_id,
            query=query,
            limit=max(
                limit * 10,
                50,
            ),
        )

        # ----------------------------------------------------
        # SCORE MÍNIMO
        # ----------------------------------------------------

        results = [
            result
            for result in results
            if float(
                result.get(
                    "exact_score",
                    1.0,
                )
            )
            >= exact_min_score
        ]

        # ----------------------------------------------------
        # DOCUMENTOS ÚNICOS
        # ----------------------------------------------------

        unique_results = []

        seen_documents = set()

        for result in results:

            document_id = result.get(
                "document_id"
            )

            if (
                document_id is None
                or
                document_id in seen_documents
            ):
                continue

            seen_documents.add(
                document_id
            )

            unique_results.append(
                result
            )

            if (
                len(unique_results)
                >= limit
            ):
                break

        results = attach_document_hashes(
            unique_results
        )

        return results

    # ========================================================
    # HYBRID
    # ========================================================

    if mode == "hybrid":

        results = hybrid_search(
            user_id=user_id,
            query=query,
            limit=limit,
            exact_percentage=exact_percentage,
            semantic_percentage=semantic_percentage,
            exact_min_score=exact_min_score,
            semantic_min_score=semantic_min_score,
        )

        results = attach_document_hashes(
            results
        )

        return results

    # ========================================================
    # MODO INVÁLIDO
    # ========================================================

    raise ValueError(
        f"Modo de búsqueda no válido: {mode}"
    )

