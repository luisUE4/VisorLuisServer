
--------------------------------------------------------------------------------
## mi_RAG 
--------------------------------------------------------------------------------
es el sistema de busqueda e indexado , intenta ser independiente del main.py 
para poder modificar y evolucionar 

ARQUITECTURA Y ALCANCE ACTUAL DEL MOTOR DE BÚSQUEDA DOCUMENTAL RAG

1. OBJETIVO

El sistema implementa un motor de búsqueda documental local para documentos PDF.

La arquitectura actual combina:

* Búsqueda semántica mediante embeddings.
* Búsqueda textual/exacta.
* Detección de identificadores.
* Búsqueda híbrida.
* Ranking combinado de resultados.

El motor está diseñado para funcionar sin depender de un LLM durante las búsquedas normales.

El objetivo de esta etapa es disponer de un buscador determinista, local, predecible y relativamente sencillo de mantener.

2. COMPONENTES PRINCIPALES

La arquitectura actual utiliza:

* FastAPI como punto de entrada de la API.
* SQLite para documentos, estados, trabajos y metadatos.
* SQLite FTS5 para búsqueda textual/exacta.
* LanceDB para embeddings y búsqueda vectorial.
* PyMuPDF para extracción de texto de PDF.
* Sentence Transformers para generación de embeddings.
* PyTorch + CUDA para aceleración mediante GPU.
* Worker Python para procesamiento de IA.

El modelo de embeddings utilizado actualmente en las pruebas es:

* intfloat/multilingual-e5-small

El modelo puede ejecutarse en CUDA cuando existe una GPU compatible.

3. TIPOS DE BÚSQUEDA

3.1 BÚSQUEDA SEMÁNTICA

La consulta original se convierte en un embedding y se compara contra los embeddings almacenados en LanceDB.

La búsqueda semántica permite encontrar contenido relacionado aunque las palabras utilizadas en la consulta no coincidan literalmente con el texto del documento.

Ejemplo:

Consulta:

"¿Cuál es el comprobante de pago de electricidad?"

Resultado esperado:

Pago_1324808388.pdf

aunque la consulta no contenga literalmente el número de folio.

La búsqueda semántica utiliza exclusivamente la consulta original.

No se modifica la consulta para generar embeddings adicionales a partir de identificadores.

3.2 BÚSQUEDA EXACTA

La búsqueda exacta utiliza SQLite FTS5.

Está orientada principalmente a:

* términos literales;
* frases;
* códigos;
* RFC;
* números de factura;
* folios;
* referencias;
* identificadores;
* otros valores donde la coincidencia literal sea importante.

3.3 DETECCIÓN DE IDENTIFICADORES

Antes de realizar la búsqueda híbrida se ejecuta:

extract_identifiers(query)

Esta función detecta posibles identificadores presentes en la consulta.

Ejemplo:

Consulta:

"¿Cuál es el comprobante de pago 1324808388?"

Resultado:

IDENTIFICADORES: ['1324808388']

La detección de identificadores actualmente es una operación determinista.

No utiliza un LLM para interpretar la consulta.

3.4 BÚSQUEDA DE IDENTIFICADORES

Los identificadores detectados se utilizan como consultas exactas adicionales mediante:

identifier_search()

Por lo tanto, una consulta que contiene un identificador genera conceptualmente:

1. búsqueda semántica de la consulta original;
2. búsqueda exacta de la consulta original;
3. búsqueda exacta adicional de cada identificador detectado.

Los identificadores son una señal adicional para el ranking.

4. BÚSQUEDA HÍBRIDA

La función principal es:

hybrid_search(
user_id: str,
query: str,
limit: int = 10
)

La función combina los resultados semánticos y exactos y posteriormente calcula un ranking común.

Flujo actual:

CONSULTA
|
+--> semantic_search(query)
|
+--> exact_search(query)
|
+--> extract_identifiers(query)
|
+--> identifier_search(identifier)
|
+--> merge de resultados
|
+--> cálculo de scores
|
+--> clasificación
|
+--> resultados finales

5. SEÑALES DE RANKING

Cada resultado puede contener las siguientes señales:

semantic_score
exact_score
exact_query_score
identifier_score
score

También se conserva:

match_type

Los tipos utilizados actualmente son:

* semantic
* exact
* hybrid

5.1 semantic_score

Representa la relevancia semántica obtenida mediante la búsqueda vectorial.

5.2 exact_query_score

Representa la coincidencia exacta con la consulta original.

5.3 identifier_score

Representa la coincidencia exacta con alguno de los identificadores detectados en la consulta.

5.4 exact_score

Es una señal derivada:

exact_score = max(
exact_query_score,
identifier_score * 0.90
)

La coincidencia de la consulta original tiene prioridad sobre la señal derivada del identificador.

6. RANKING HÍBRIDO ACTUAL

Cuando existe coincidencia con un identificador:

score =
0.80 * identifier_score
+
0.15 * exact_query_score
+
0.05 * semantic_score

Esto proporciona una prioridad muy alta a una coincidencia exacta del identificador.

Cuando existe coincidencia exacta con la consulta original pero no con un identificador:

score =
0.75 * exact_query_score
+
0.25 * semantic_score

Cuando no existe coincidencia exacta:

score =
0.65 * semantic_score
+
0.35 * exact_score

El objetivo no es construir un ranking matemáticamente perfecto, sino establecer una jerarquía clara entre las diferentes señales disponibles.

7. PRIORIDAD DE IDENTIFICADORES

Los identificadores tienen una prioridad especial porque normalmente representan valores con significado literal.

Ejemplos:

* número de factura;
* folio;
* referencia bancaria;
* RFC;
* número de comprobante;
* código;
* identificador documental.

Si un documento contiene exactamente el identificador solicitado, esa coincidencia debe tener prioridad sobre un resultado que únicamente sea semánticamente parecido.

Ejemplo:

Consulta:

"¿Cuál es el comprobante de pago 1324808388?"

Documento correcto:

Pago_1324808388.pdf

Señales observadas:

identifier_score = 1.0
semantic_score = 1.0
exact_score = 0.9
match_type = hybrid
score = 0.85

Esto demuestra que el documento que contiene el identificador recibe una prioridad muy alta.

8. COMPORTAMIENTO CUANDO EL IDENTIFICADOR NO EXISTE

Actualmente el sistema NO interpreta que la presencia de un identificador necesariamente constituye una restricción lógica que deba cumplirse.

Ejemplo:

"¿Cuál es el comprobante de pago 9999999999?"

Si 9999999999 no existe en ningún documento, actualmente puede aparecer un documento semánticamente relacionado, por ejemplo:

Pago_1324808388.pdf

con:

identifier_score = 0.0
exact_query_score = 0.0
semantic_score = 1.0
match_type = semantic
score = 0.65

Este comportamiento es intencional dentro del alcance actual.

El sistema reconoce que existe un identificador, pero todavía no interpreta la intención del usuario asociada a dicho identificador.

9. LÍMITES ACTUALES DE INTERPRETACIÓN

La implementación actual NO intenta comprender operadores semánticos o intenciones complejas sobre identificadores.

Por ejemplo, actualmente no existe una capa específica para interpretar correctamente diferencias entre:

"busca el documento 1324808388"

"busca documentos relacionados con 1324808388"

"busca documentos que no tengan 1324808388"

"busca todos excepto 1324808388"

"busca pagos distintos del 1324808388"

Estas expresiones requieren interpretar la intención del usuario, no solamente detectar el identificador.

Esta capacidad queda fuera del alcance de la implementación actual.

No se deben añadir reglas complejas al motor híbrido únicamente para intentar simular comprensión de lenguaje natural.

10. PRINCIPIO ARQUITECTÓNICO

La búsqueda actual debe permanecer independiente de un LLM.

El motor actual debe continuar funcionando como:

consulta
|
+--> búsqueda exacta
+--> detección de identificadores
+--> búsqueda semántica
+--> ranking híbrido
|
+--> resultados

El LLM no forma parte del funcionamiento necesario para realizar una búsqueda normal.

Esto mantiene el sistema:

* local;
* determinista;
* explicable;
* predecible;
* sencillo de mantener;
* independiente de servicios externos de IA.

11. EJEMPLOS DE COMPORTAMIENTO ACTUAL

EJEMPLO 1 — IDENTIFICADOR EXISTENTE

Consulta:

"¿Cuál es el comprobante de pago 1324808388?"

Identificadores detectados:

['1324808388']

Resultado principal esperado:

Pago_1324808388.pdf

Señales aproximadas observadas:

semantic_score = 1.0
identifier_score = 1.0
exact_score = 0.9
match_type = hybrid
score = 0.85

Interpretación:

El documento contiene literalmente el identificador solicitado y además es semánticamente relevante.

---

EJEMPLO 2 — IDENTIFICADOR INEXISTENTE

Consulta:

"¿Cuál es el comprobante de pago 9999999999?"

Identificadores detectados:

['9999999999']

No existe coincidencia exacta del identificador.

Actualmente pueden aparecer resultados semánticamente relacionados.

Ejemplo:

Pago_1324808388.pdf

con:

identifier_score = 0.0
exact_query_score = 0.0
semantic_score = 1.0
match_type = semantic

Interpretación:

El motor encontró contenido semánticamente relacionado, pero NO significa que haya encontrado el identificador solicitado.

La interpretación de esta situación queda fuera del alcance actual.

---

EJEMPLO 3 — CONSULTA SEMÁNTICA SIN IDENTIFICADOR

Consulta:

"¿Cuál es el comprobante de pago de electricidad?"

Identificadores detectados:

[]

Resultado principal esperado:

Pago_1324808388.pdf

Señales observadas:

semantic_score = 1.0
identifier_score = 0.0
exact_query_score = 0.0
match_type = semantic
score = 0.65

Interpretación:

La búsqueda funciona como búsqueda semántica normal porque no existe un identificador detectable en la consulta.

12. METADATOS DE LOS RESULTADOS

Cada resultado conserva información suficiente para localizar el contenido original.

Ejemplo conceptual:

embedding
document_id
user_id
filename
page
line_start
line_end
text

Esto permite devolver directamente:

* documento;
* página;
* rango de líneas;
* fragmento de texto.

No es necesario realizar una segunda búsqueda costosa para reconstruir el contexto del resultado.

13. EVOLUCIÓN FUTURA: LLM

En una fase posterior se podrá incorporar un LLM como una capa de interpretación de consultas.

El objetivo NO será reemplazar el motor de búsqueda actual.

La arquitectura futura podría ser:

CONSULTA DEL USUARIO
|
v
QUERY INTERPRETATION
|
v
CONSULTA ESTRUCTURADA
|
v
MOTOR DE BÚSQUEDA ACTUAL
|
+--> SQLite / FTS5
+--> identificadores
+--> LanceDB
|
v
RESULTADOS

El LLM podrá interpretar intenciones que actualmente están fuera del alcance del sistema, por ejemplo:

* inclusión de identificadores;
* exclusión de identificadores;
* negaciones;
* filtros;
* combinaciones de condiciones;
* consultas más complejas en lenguaje natural.

El LLM debe interpretar la intención y producir una representación estructurada.

La ejecución real de la búsqueda debe continuar realizándose mediante los motores deterministas existentes siempre que sea posible.

Principio futuro:

LLM = interpretación

SQLite / FTS5 / LanceDB = ejecución de búsqueda

14. EVOLUCIÓN FUTURA: OCR

Actualmente la extracción de texto depende de que el PDF contenga texto extraíble.

Una futura fase incorporará OCR para documentos escaneados o imágenes que no contengan una capa de texto utilizable.

Flujo futuro:

PDF
|
+--> ¿tiene texto extraíble?
|       |
|       +--> SÍ --> extracción normal
|       |
|       +--> NO --> OCR
|
+--> chunking
|
+--> embeddings
|
+--> indexación
|
+--> búsqueda

El OCR permitirá ampliar el sistema a:

* PDFs escaneados;
* documentos fotografiados;
* imágenes dentro de documentos;
* comprobantes sin capa de texto;
* documentos donde la extracción convencional produzca poco o ningún texto.

La incorporación de OCR no debe modificar innecesariamente la arquitectura de búsqueda. Su función principal será proporcionar texto utilizable para las etapas posteriores de chunking, búsqueda exacta y generación de embeddings.

15. EVOLUCIÓN GENERAL DEL SISTEMA

ETAPA ACTUAL:

PDF
|
v
extracción de texto
|
v
chunking
|
+--> SQLite / FTS5
|
+--> embeddings
|
v
LanceDB
|
v
búsqueda híbrida

ETAPA FUTURA:

PDF / imagen
|
v
extracción de texto
|
+--> OCR cuando sea necesario
|
v
chunking
|
+--> búsqueda exacta
|
+--> embeddings
|
+--> búsqueda semántica
|
+--> búsqueda híbrida
|
v
interpretación avanzada mediante LLM
cuando sea necesaria

16. PRINCIPIOS QUE DEBEN CONSERVARSE

17. La búsqueda normal no debe depender de un LLM.

18. La búsqueda exacta debe continuar utilizando mecanismos deterministas.

19. Los identificadores deben conservarse como una señal específica de ranking.

20. Una coincidencia literal de un identificador debe tener una prioridad alta frente a una coincidencia únicamente semántica.

21. El modelo de embeddings debe ser configurable.

22. Los embeddings deben conservar metadatos suficientes para localizar el texto original.

23. SQLite y LanceDB tienen responsabilidades distintas.

24. El worker controla el procesamiento pesado y el uso de GPU.

25. La incorporación futura de un LLM debe añadir capacidades de interpretación, no sustituir innecesariamente el motor de búsqueda.

26. OCR debe incorporarse como una etapa de adquisición de texto cuando la extracción convencional no sea suficiente.

27. No se deben añadir reglas de lenguaje natural excesivamente complejas al buscador determinista para resolver problemas que pertenecen a la futura capa de interpretación.

28. Las limitaciones actuales deben considerarse parte del diseño y no necesariamente errores de implementación.

29. ESTADO ACTUAL

La implementación actual de búsqueda híbrida se considera funcional y razonable para la primera etapa del sistema.

Las capacidades actualmente implementadas son suficientes para:

* búsqueda semántica;
* búsqueda exacta;
* búsqueda por identificadores;
* combinación de resultados;
* ranking híbrido;
* priorización de coincidencias literales de identificadores.

La interpretación avanzada de intención y el procesamiento OCR quedan definidos como evoluciones posteriores.

El objetivo de esta separación es evitar sobrecargar la implementación actual y mantener un motor de búsqueda local, determinista y fácil de mantener mientras se prepara una arquitectura capaz de incorporar posteriormente capacidades de LLM y OCR.

