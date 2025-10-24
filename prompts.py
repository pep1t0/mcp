"""
Templates de prompts y modelos Pydantic para el agente MCP.
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


# ==========================
# 📦 MODELOS PYDANTIC
# ==========================

class ToolDecision(BaseModel):
    """Decisión sobre qué herramienta usar."""
    needs_tool: bool = Field(description="Si necesita usar una herramienta")
    tool: str = Field(default="", description="Nombre de la herramienta (si needs_tool=True)")
    arguments: dict = Field(default={}, description="Argumentos de la herramienta")
    reasoning: str = Field(description="Por qué usa (o no) la herramienta")


class Response(BaseModel):
    """Respuesta final al usuario."""
    answer: str = Field(description="Respuesta en lenguaje natural")


class GoalEvaluation(BaseModel):
    """Evaluación del progreso del objetivo."""
    status: Literal["completed", "in_progress", "failed"] = Field(
        description="Estado del objetivo: 'completed' si está completo, 'in_progress' si puede continuar, 'failed' si no puede continuar y no está completo"
    )
    next_step: str = Field(default="", description="Siguiente paso a realizar (si status='in_progress')")
    reasoning: str = Field(description="Razonamiento de la evaluación")
    no_tools_available: bool = Field(
        default=False,
        description="Marca como True si NO existen herramientas disponibles para completar el objetivo. Esto finalizará la ejecución inmediatamente."
    )


# ==========================
# 📋 TEMPLATES DE PROMPTS
# ==========================

def create_decision_template(tools_description: str) -> ChatPromptTemplate:
    """
    Crea un template de decisión dinámico con las herramientas disponibles.
    
    Args:
        tools_description: String con la descripción de todas las herramientas disponibles
    
    Returns:
        ChatPromptTemplate configurado con las herramientas
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""Eres un asistente que decide QUÉ herramienta usar para responder.

HERRAMIENTAS DISPONIBLES:
{tools_description}

REGLAS OBLIGATORIAS:
1. Si la tarea dice "Usar <nombre_herramienta>" → DEBES marcar needs_tool=True y especificar tool=<nombre_herramienta>
2. Si necesitas información del sistema (usuario, disco, archivos, red) → DEBES usar una herramienta (needs_tool=True)
3. Solo marca needs_tool=False si la pregunta es teórica o ya tienes toda la información

GUÍA PARA BÚSQUEDAS EN OPENSEARCH (vector_search vs text_search vs hybrid_search):

USA **vector_search** CUANDO:
- Buscas por CONCEPTO o SIGNIFICADO (ej: "incidentes de ransomware", "amenazas críticas")
- Quieres encontrar contenido SIMILAR aunque no contenga las palabras exactas
- Buscas por CONTEXTO semántico (ej: "ataques a empresas textiles")
- Es una búsqueda EXPLORATORIA o abierta

USA **text_search** CUANDO:
- Buscas PALABRAS EXACTAS o nombres específicos (ej: "MANGO", "CVE-2024-1234")
- Necesitas coincidencias LITERALES
- Buscas en campos específicos conocidos
- Es una búsqueda PRECISA de términos concretos

USA **hybrid_search** CUANDO:
- Quieres COMBINAR precisión léxica + contexto semántico
- No estás seguro de qué tipo de búsqueda es mejor
- Quieres resultados MÁS COMPLETOS (pero más lentos)

RECOMENDACIÓN POR DEFECTO: 
- Para búsquedas de seguridad/ciberinteligencia → **vector_search** (mejor recall)
- Para búsquedas de nombres propios/IDs → **text_search** (mejor precisión)
- Si no estás seguro → **hybrid_search** (mejor balance)

FORMATO DE ARGUMENTOS:
- Parámetros tipo 'dict' o 'object' → USA objetos JSON directamente, NO strings
- Parámetros tipo 'list' o 'array' → USA arrays [], NO strings
- Parámetros opcionales que no necesitas → USA null o no los incluyas

Ejemplos CORRECTOS de formato de arguments:
- Para vector_search: index_name como string, query como string, top_k como número
- Para text_search: index_name como string, query_text como string, fields como array de strings
- Para hybrid_search: index_name como string, query_text como string, vector_weight como número decimal

Ejemplos INCORRECTOS:
- arguments con fields como string en vez de lista
- arguments con filter_query como string en vez de objeto JSON

FORMATO DE RESPUESTA:
- needs_tool: true/false
- tool: nombre exacto de la herramienta (ej: "vector_search", "text_search")
- arguments: diccionario con los parámetros (tipos correctos: strings, números, listas, objetos)
- reasoning: breve explicación de POR QUÉ elegiste esa herramienta de búsqueda"""),
        ("human", "{question}")
    ])

def create_evaluation_template(tools_description: str) -> ChatPromptTemplate:
    """
    Crea un template de evaluación dinámico con las herramientas disponibles.
    
    Args:
        tools_description: String con la descripción de todas las herramientas disponibles
    
    Returns:
        ChatPromptTemplate configurado con las herramientas
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""Evalúa el progreso hacia un objetivo complejo.

HERRAMIENTAS DISPONIBLES:
{tools_description}

ESTADOS POSIBLES:
- 'completed': El objetivo está 100% cumplido con resultados REALES obtenidos de herramientas ejecutadas
- 'in_progress': Aún necesitas ejecutar herramientas para obtener información (especifica next_step)
- 'failed': No puedes continuar (sin herramientas útiles, bloqueado, imposible)

CAMPO ESPECIAL - no_tools_available:
- Marca como **True** si NINGUNA de las herramientas disponibles puede ayudar con el objetivo
- Esto finalizará la ejecución INMEDIATAMENTE, evitando iteraciones innecesarias
- Usa esto cuando el objetivo requiere capacidades que NO existen en las herramientas
- Ejemplos: preguntas sobre tu identidad, modelo de IA, capacidades internas, filosofía, etc.

REGLAS CRÍTICAS:
1. Si el historial está VACÍO o el último resultado es "Comenzando..." → SIEMPRE marca 'in_progress'
2. Solo marca 'completed' si YA ejecutaste herramientas y tienes resultados reales
3. Si necesitas información del sistema → marca 'in_progress' y especifica qué herramienta usar
4. **IMPORTANTE**: Si necesitas descargar múltiples archivos, especifica UNO SOLO por iteración en next_step
5. El next_step debe ser muy específico: incluye el nombre exacto de la herramienta Y sus parámetros
6. Si no_tools_available=True → Explica en 'reasoning' POR QUÉ ninguna herramienta puede ayudar

Ejemplo:
- Objetivo: "dime qué usuario estoy usando"
- Historial: []
- Último resultado: "Comenzando..."
→ Estado: 'in_progress', next_step: 'Usar get_current_user para obtener el nombre del usuario'

Ejemplo con parámetros:
- Objetivo: "descargar archivos del FTP"
- Historial: [lista con flag.txt y reddit.png]
- Último resultado: "Archivos encontrados: flag.txt, reddit.png"
→ Estado: 'in_progress', next_step: 'Usar ftp_download_file(ip=192.168.0.248, filename=flag.txt) para descargar el primer archivo', no_tools_available: False

Ejemplo sin herramientas disponibles:
- Objetivo: "¿Qué modelo de IA eres?"
- Historial: []
- Último resultado: "Comenzando..."
→ Estado: 'in_progress', next_step: '', reasoning: 'Ninguna herramienta disponible permite introspección del modelo de IA', no_tools_available: True"""),
        ("human", """Objetivo del usuario: {goal}

Historial de acciones:
{history}

Último resultado: {last_result}

¿Cuál es el estado del objetivo y qué debería hacer a continuación?""")
    ])


RESPONSE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que responde de forma clara y concisa."),
    ("human", """Pregunta: {question}
    
Resultado de la herramienta: {tool_result}

Genera una respuesta natural para el usuario.""")
])



# Template estático (deprecated - usar create_evaluation_template)
GOAL_EVALUATION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Evalúa el progreso hacia un objetivo complejo.

ESTADOS POSIBLES:
- 'completed': El objetivo está completamente cumplido
- 'in_progress': El objetivo no está completo pero puedes seguir trabajando (especifica next_step)
- 'failed': El objetivo no está completo Y no puedes continuar (sin herramientas útiles, bloqueado, imposible)

Sé honesto: si no tienes herramientas para lograr algo, marca como 'failed'."""),
    ("human", """Objetivo del usuario: {goal}

Historial de acciones:
{history}

Último resultado: {last_result}

¿Cuál es el estado del objetivo y qué debería hacer a continuación?""")
])