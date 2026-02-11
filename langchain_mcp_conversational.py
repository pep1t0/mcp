"""
Agente MCP implementado con LangGraph para flujos más robustos.
"""
import asyncio
import json
from enum import Enum
from typing import TypedDict, Annotated, Literal, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.exceptions import OutputParserException
from prompts import (
    ToolDecision, 
    Response, 
    GoalEvaluation, 
    create_decision_template,
    create_evaluation_template,
    RESPONSE_TEMPLATE
)


# ==========================
# 📦 MODELOS DE DATOS
# ==========================

class AgentStatus(Enum):
    """Estados posibles del agente para routing explícito."""
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    NO_TOOLS = "no_tools"


class HistoryEntry(TypedDict):
    """Entrada individual en el historial de acciones ejecutadas por el agente."""
    step: str
    tool: str
    args: dict[str, Any]
    result: str


class AgentState(TypedDict):
    """Estado que se pasa entre nodos del grafo."""
    goal: str
    history: Annotated[
        list[HistoryEntry], 
        "Historial de acciones ejecutadas (últimas 10 para gestión de contexto)"
    ]
    full_history: Annotated[
        list[HistoryEntry],
        "Historial completo sin límites para logs"
    ]
    current_step: str
    last_result: str
    completed: bool
    can_continue: bool
    iteration: int
    max_iterations: int
    final_answer: str
    tool_decision: dict[str, Any]  # Decisión de qué herramienta usar
    status: AgentStatus  # Estado explícito del agente


# ==========================
# 🧩 NODOS DEL GRAFO
# ==========================

# Constantes
MAX_HISTORY_WINDOW = 10  # Ventana deslizante de historial para contexto LLM

class GraphAgent:
    def __init__(self, llm, client):
        self.llm = llm
        self.client = client
        self.response_chain = RESPONSE_TEMPLATE | llm.with_structured_output(Response)
        # decision_chain y evaluation_chain se crean dinámicamente en initialize()
        self.decision_chain = None
        self.evaluation_chain = None
        self.available_tools = []  # Lista de nombres de herramientas disponibles
        self.tools_cache = {}  # Cache de herramientas por nombre
    
    def _format_history_for_llm(self, history: list[HistoryEntry]) -> str:
        """Formatea el historial para el LLM con ventana deslizante."""
        if not history:
            return "Sin acciones previas"
        
        # Solo últimas MAX_HISTORY_WINDOW entradas
        recent_history = history[-MAX_HISTORY_WINDOW:]
        formatted = []
        for entry in recent_history:
            formatted.append(
                f"Step: {entry['step']}\n"
                f"Tool: {entry['tool']}\n"
                f"Args: {entry['args']}\n"
                f"Result: {entry['result'][:500]}...\n"  # Truncar resultados largos
            )
        return "\n".join(formatted)
    
    def _validate_tool_decision(self, decision: ToolDecision, state: AgentState) -> ToolDecision:
        """Valida y corrige la decisión del LLM en código Python."""
        # Normalizar nombre de herramienta a str (evitar None raros del parser)
        decision.tool = decision.tool or ""

        # Validación 1: needs_tool=True requiere tool
        if decision.needs_tool and not decision.tool:
            # Intentar extraer tool del current_step
            next_step_lower = state['current_step'].lower()
            sorted_tools = sorted(self.available_tools, key=len, reverse=True)
            
            for tool_name in sorted_tools:
                if tool_name in next_step_lower:
                    print(f"⚠️  Corrección automática: Detectado '{tool_name}' en el paso actual")
                    decision.tool = tool_name
                    break
            
            # Si aún no hay tool, marcar como no necesita
            if not decision.tool:
                print(f"⚠️  Corrección: needs_tool=True pero sin tool especificada. Marcando needs_tool=False")
                decision.needs_tool = False
        
        # Validación 2: si se menciona una tool inexistente, no forzar ejecución
        if decision.tool and decision.tool not in self.available_tools:
            print(f"❌ Error: Tool '{decision.tool}' no existe. Tools disponibles: {self.available_tools}")
            decision.needs_tool = False
            decision.tool = ""
            decision.arguments = {}
        
        # Validación 3: Si queda una tool válida pero needs_tool=False, no forzamos a True;
        # dejamos que el agente continúe sin ejecutar herramienta en este paso.
        
        return decision
    
    def _validate_evaluation(self, evaluation: GoalEvaluation) -> GoalEvaluation:
        """Valida y normaliza la evaluación del LLM."""
        # Normalizar status
        valid_statuses = ["completed", "in_progress", "failed"]
        if evaluation.status not in valid_statuses:
            print(f"⚠️  Corrección: Status '{evaluation.status}' inválido. Usando 'in_progress'")
            evaluation.status = "in_progress"
        
        # Si no_tools_available=True, debe ser completed o failed
        if evaluation.no_tools_available and evaluation.status == "in_progress":
            print(f"⚠️  Corrección: no_tools_available=True requiere status 'completed' o 'failed'")
            evaluation.status = "failed"
        
        return evaluation
    
    async def initialize(self):
        """Descubre las herramientas disponibles y configura el prompt dinámicamente"""
        print("🔍 Descubriendo herramientas disponibles en servidores MCP...")
        
        # Obtener herramientas de todos los servidores MCP
        tools_info = await self.client.get_tools()

        # Cachear las herramientas por nombre
        self.tools_cache = {tool.name: tool for tool in tools_info}
        self.available_tools = list(self.tools_cache.keys())
        
        # Formatear la información de las herramientas
        tools_description = ""
        for tool in tools_info:
            name = tool.name
            # Escapar llaves en la descripción para evitar conflictos con templates de LangChain
            description = (tool.description or 'No description').replace('{', '{{').replace('}', '}}')
            
            # Formatear parámetros desde el schema de args
            params = []
            if hasattr(tool, 'args') and tool.args:
                # tool.args es un dict con la información de los parámetros
                for param_name, param_info in tool.args.items():
                    param_type = param_info.get('type', 'string')
                    # Escapar llaves en las descripciones de parámetros también
                    param_desc = param_info.get('description', '').replace('{', '{{').replace('}', '}}')
                    params.append(f"  - {param_name} ({param_type}): {param_desc}")
            
            params_str = "\n".join(params) if params else "  Sin parámetros"
            
            tools_description += f"""
Tool: {name}
Descripción: {description}
Parámetros:
{params_str}

"""
        
        print(f"✅ Descubiertas {len(tools_info)} herramientas")
        # print(f"\n{tools_description}")
        
        # Guardar nombres de herramientas para usar en correcciones
        self.available_tools = [tool.name for tool in tools_info]
        print(f"🛠️ Herramientas disponibles: {self.available_tools}\n")
        
        # Crear los prompts dinámicamente con las herramientas descubiertas
        decision_template = create_decision_template(tools_description)
        # Retry logic: 3 intentos con backoff exponencial para decision
        self.decision_chain = (
            decision_template | 
            self.llm.with_structured_output(ToolDecision, include_raw=True)
        )
        
        evaluation_template = create_evaluation_template(tools_description)
        # Retry logic: 3 intentos con backoff exponencial para evaluation
        self.evaluation_chain = (
            evaluation_template | 
            self.llm.with_structured_output(GoalEvaluation, include_raw=True)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OutputParserException)
    )
    async def evaluate_node(self, state: AgentState) -> AgentState:
        """Evalúa el progreso y decide el siguiente paso con retry logic."""
        print(f"\n--- Iteración {state['iteration']}/{state['max_iterations']} ---")
        print("📊 Evaluando progreso...")
        
        # Usar ventana deslizante de historial
        recent_history = self._format_history_for_llm(state["history"])
        
        try:
            evaluation_result = await self.evaluation_chain.ainvoke({
                "goal": state["goal"],
                "history": recent_history,
                "last_result": state["last_result"]
            })
            
            # Extraer parsed result (puede ser dict o objeto)
            if isinstance(evaluation_result, dict) and "parsed" in evaluation_result:
                evaluation = evaluation_result["parsed"]
            else:
                evaluation = evaluation_result
            
        except OutputParserException as e:
            print(f"⚠️  Error de parsing en evaluación (intento con retry): {e}")
            # Dejar que tenacity lo reintente
            raise
        except Exception as e:
            print(f"❌ Error inesperado en evaluación: {e}")
            # Crear evaluación por defecto para continuar
            evaluation = GoalEvaluation(
                status="failed",
                next_step="",
                reasoning=f"Error al evaluar: {str(e)}",
                no_tools_available=False
            )
        
        # ✅ VALIDACIÓN EN CÓDIGO (no en template)
        evaluation = self._validate_evaluation(evaluation)
        
        print(f"🧠 Evaluación: {evaluation.reasoning}")
        print(f"📌 Estado: {evaluation.status}")
        
        # Mapear status a AgentStatus enum
        if evaluation.no_tools_available:
            # Caso explícito: el LLM indica que no hay herramientas útiles.
            state["status"] = AgentStatus.NO_TOOLS
            state["completed"] = True
            state["can_continue"] = False
            state["last_result"] = evaluation.reasoning
        elif evaluation.status == "completed":
            state["status"] = AgentStatus.COMPLETED
            state["completed"] = True
            state["can_continue"] = False
            # ✅ Preservar contexto para finalize_node
            state["last_result"] = evaluation.reasoning
        elif evaluation.status == "failed":
            state["status"] = AgentStatus.FAILED
            state["completed"] = False
            state["can_continue"] = False
        else:  # in_progress
            state["status"] = AgentStatus.EXECUTING
            state["completed"] = False
            state["can_continue"] = True
        
        state["current_step"] = evaluation.next_step

        # Solo incrementamos iteración mientras siga en progreso
        if state["status"] == AgentStatus.EXECUTING:
            state["iteration"] += 1
            # Detección de límite de iteraciones solo para estados no terminales
            if state["iteration"] >= state["max_iterations"]:
                state["status"] = AgentStatus.MAX_ITERATIONS
                state["can_continue"] = False
                print(f"⚠️  Alcanzado límite de {state['max_iterations']} iteraciones")
        
        return state
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OutputParserException)
    )
    async def decide_node(self, state: AgentState) -> AgentState:
        """Decide qué herramienta usar con retry logic y validación."""
        print(f"➡️  Siguiente paso: {state['current_step']}")
        
        try:
            decision_result = await self.decision_chain.ainvoke({
                "question": state["current_step"]
            })
            
            # Extraer parsed result
            if isinstance(decision_result, dict) and "parsed" in decision_result:
                decision = decision_result["parsed"]
            else:
                decision = decision_result
            
        except OutputParserException as e:
            print(f"⚠️  Error de parsing en decisión (intento con retry): {e}")
            raise
        except Exception as e:
            print(f"❌ Error inesperado en decisión: {e}")
            state["can_continue"] = False
            state["last_result"] = f"Error al decidir herramienta: {str(e)}"
            return state
        
        # ✅ VALIDACIÓN EN CÓDIGO (no heurísticas en template)
        decision = self._validate_tool_decision(decision, state)
        
        print(f"🧠 Razonamiento: {decision.reasoning}")
        print(f"🔧 Tool: {decision.tool if decision.tool else 'Ninguna'}")
        print(f"❓ Needs tool: {decision.needs_tool}")
        
        state["tool_decision"] = {
            "needs_tool": decision.needs_tool,
            "tool": decision.tool,
            "arguments": decision.arguments
        }
        
        return state
    
    async def execute_node(self, state: AgentState) -> AgentState:
        """Ejecuta la herramienta decidida."""
        decision = state.get("tool_decision", {})
        
        if not decision.get("needs_tool"):
            print("⚠️  No se necesita herramienta")
            state["can_continue"] = False
            return state
        
        tool_name = decision["tool"]
        args = decision["arguments"]
        
        # ✅ Validar IDs antes de get_documents_by_ids
        if tool_name == "get_documents_by_ids":
            ids = args.get("ids", [])
            if not ids or not all(isinstance(i, str) and i.strip() for i in ids):
                result_str = "Error: IDs vacíos o inválidos. Usar IDs completos de búsquedas previas."
                print(f"❌ {result_str}")
                history_entry: HistoryEntry = {
                    "step": state["current_step"],
                    "tool": tool_name,
                    "args": args,
                    "result": result_str
                }
                state["full_history"].append(history_entry)
                state["history"].append(history_entry)
                state["history"] = state["history"][-MAX_HISTORY_WINDOW:]
                state["last_result"] = result_str
                return state
        
        print(f"⚙️  Ejecutando: {tool_name}({args})")
        
        try:
            # Buscar la herramienta por nombre en las tools disponibles
            tool_obj = self.tools_cache.get(tool_name)
            if not tool_obj:
                tools = await self.client.get_tools()
                tool_obj = next((t for t in tools if t.name == tool_name), None)

            if not tool_obj:
                result_str = f"Error: Herramienta '{tool_name}' no encontrada"
                print(f"❌ {result_str}")
            else:
                # Invocar la herramienta usando LangChain
                result = await tool_obj.ainvoke(args)
                
                # ✅ NUEVO: Extracción inteligente para verify_and_extract
                try:
                    result_json = json.loads(str(result))
                    
                    if tool_name == "verify_and_extract":
                        # Crear resumen estructurado priorizando links
                        summary = {
                            "url": result_json.get("url"),
                            "title": result_json.get("title", "Sin título"),
                            "status": result_json.get("status_code"),
                            "accessible": result_json.get("is_accessible", False),
                            "text_preview": result_json.get("text", "")[:1000],  # Primer 1KB del texto
                            "links": result_json.get("links", [])[:15]  # ✅ Primeros 15 enlaces con contexto
                        }
                        result_str = json.dumps(summary, ensure_ascii=False, indent=2)
                        print(f"✅ Resultado (con links extraídos): {len(result_json.get('links', []))} enlaces encontrados")
                    else:
                        # Otras herramientas: truncar normalmente
                        result_str = str(result)[:3000]
                except (json.JSONDecodeError, TypeError):
                    # Si no es JSON o falla el parsing, truncar normalmente
                    result_str = str(result)[:3000]
                
                print(f"✅ Resultado: {result_str[:500]}...")  # Mostrar preview en consola
        except Exception as e:
            result_str = f"Error: {str(e)}"
            print(f"❌ {result_str}")
        
        history_entry = {
            "step": state["current_step"],
            "tool": tool_name,
            "args": args,
            "result": result_str
        }
        
        # Agregar a full_history (sin límites)
        state["full_history"].append(history_entry)
        
        # Agregar a history con ventana deslizante
        state["history"].append(history_entry)
        if len(state["history"]) > MAX_HISTORY_WINDOW:
            state["history"] = state["history"][-MAX_HISTORY_WINDOW:]
        
        state["last_result"] = result_str
        
        return state
    
    async def finalize_node(self, state: AgentState) -> AgentState:
        """Genera la respuesta final."""
        print(f"\n{'='*60}")
        print("🎉 ¡OBJETIVO COMPLETADO!")
        print(f"{'='*60}\n")
        
        # ✅ Si no hay historial (no se ejecutaron herramientas), usar last_result directamente
        if not state["full_history"]:
            # El LLM ya explicó por qué no puede responder en last_result
            # state["final_answer"] = state["last_result"]
            
            # En lugar de usar last_result (razonamiento técnico), generar respuesta natural
            social_response_prompt = f"""
            El usuario dijo: "{state['goal']}"
            
            Esta es una interacción que no requiere herramientas.
            Genera una respuesta natural, amigable y cortés en español teniendo presente lo que
            el usuario escribó {state['goal']} así como la reflexión interna del asistente: {state['last_result']}.

            Respuesta:
            """
            
            # Usar el LLM para generar respuesta natural
            response = await self.llm.ainvoke(social_response_prompt)
            state["final_answer"] = response.content.strip()
        else:
            # Si hay historial, generar resumen de las acciones realizadas
            response: Response = await self.response_chain.ainvoke({
                "question": f"Resume lo que hiciste para cumplir: {state['goal']}",
                "tool_result": str(state["full_history"])
            })
            state["final_answer"] = response.answer
        
        return state


# ==========================
# 🔀 LÓGICA DE ROUTING
# ==========================

def should_continue(state: AgentState) -> Literal["decide", "finalize", "end"]:
    """Decide el siguiente nodo con routing explícito basado en AgentStatus."""
    
    status = state.get("status", AgentStatus.EXECUTING)
    
    # Si ya no podemos continuar, siempre intentamos pasar por finalize_node
    # para devolver alguna respuesta al usuario (aunque sea parcial).
    if not state.get("can_continue", True) or status in [
        AgentStatus.COMPLETED,
        AgentStatus.NO_TOOLS,
        AgentStatus.FAILED,
        AgentStatus.MAX_ITERATIONS,
    ]:
        return "finalize"
    
    # Verificación adicional de límites de seguridad
    if state.get("iteration", 0) >= state.get("max_iterations", 0):
        return "finalize"
    
    # Continuar ejecutando
    return "decide"


# ==========================
# 🏗️ CONSTRUCCIÓN DEL GRAFO
# ==========================

async def create_agent_graph(llm, client):
    """Crea el grafo del agente."""
    agent = GraphAgent(llm, client)
    
    # Inicializar: descubrir herramientas disponibles
    await agent.initialize()
    
    # Crear el grafo
    workflow = StateGraph(AgentState)
    
    # Agregar nodos
    workflow.add_node("evaluate", agent.evaluate_node)
    workflow.add_node("decide", agent.decide_node)
    workflow.add_node("execute", agent.execute_node)
    workflow.add_node("finalize", agent.finalize_node)
    
    # Definir el flujo
    workflow.set_entry_point("evaluate")
    
    # Edges condicionales desde evaluate
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "decide": "decide",
            "finalize": "finalize",
            "end": END
        }
    )
    
    # Edge de decide a execute
    workflow.add_edge("decide", "execute")
    
    # Edge de execute de vuelta a evaluate (loop)
    workflow.add_edge("execute", "evaluate")
    
    # Edge de finalize al END
    workflow.add_edge("finalize", END)
    
    # ✅ Aumentar recursion_limit para evitar errores prematuros
    return workflow.compile(
        checkpointer=None,
        debug=False  # Cambiar a True para ver el flujo completo del grafo
    )


# ==========================
# 🎮 MAIN
# ==========================

async def main():
    llm = ChatOllama(
        model="qwen3:8b", 
        base_url="http://localhost:11434", 
        temperature=0,
        timeout=120  # 2 minutos de timeout
    )
    
    client = MultiServerMCPClient({
        "shell": {
            "command": "python",
            "args": ["/Users/dani/Projectes/mcp/servers/shell_mcp_server_local.py"],
            "transport": "stdio",
        },
        # "nmap": {
        #     "transport": "streamable_http",
        #     "url": "http://172.16.207.128:8080/mcp",  # URL del servidor nmap
        # },        
        "opensearch": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",  # URL del servidor opensearch_mcp_server.py
        },
        "internet": {
            "command": "python",
            "args": ["/Users/dani/Projectes/mcp/servers/internet_mcp_server.py"],
            "transport": "stdio",
        }
    })
    
    # Crear el grafo UNA SOLA VEZ antes del loop
    print("🔧 Inicializando agente...")
    graph = await create_agent_graph(llm, client)
    
    # ✅ NUEVO: Historial persistente a nivel de sesión
    session_history: list[HistoryEntry] = []
    
    print("="*60)
    print("🤖 AGENTE MCP con LangGraph")
    print("="*60)
    print("📝 Escribe un objetivo (o 'salir' para terminar)")
    print("="*60)
    print()
    
    while True:
        user_input = input("💬 Objetivo: ").strip()
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("👋 ¡Hasta luego!")
            break
        
        if not user_input:
            continue
        
        # Estado inicial con historial de sesión
        initial_state: AgentState = {
            "goal": user_input,
            "history": session_history[-MAX_HISTORY_WINDOW:],  # ✅ Últimas N entradas de la sesión
            "full_history": session_history.copy(),             # ✅ Todo el historial de la sesión
            "current_step": "",
            "last_result": "Comenzando..." if not session_history else "Continuando sesión...",
            "completed": False,
            "can_continue": True,
            "iteration": 0,
            "max_iterations": 7,  # ✅ Reducido de 15 a 7 para mayor eficiencia
            "final_answer": "",
            "tool_decision": {},
            "status": AgentStatus.PLANNING
        }
        
        # Ejecutar el grafo (reutilizado)
        try:
            final_state = await graph.ainvoke(initial_state)
            
            # ✅ NUEVO: Actualizar historial de sesión con nuevas entradas
            new_entries = final_state.get("full_history", [])[len(session_history):]
            session_history.extend(new_entries)
            
            if final_state.get("final_answer"):
                print(f"\n✅ {final_state['final_answer']}\n")
            else:
                print(f"\n⚠️  No se pudo completar el objetivo\n")
        except Exception as e:
            print(f"\n❌ Error al ejecutar: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())