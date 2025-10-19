"""
Agente MCP implementado con LangGraph para flujos más robustos.
Versión optimizada para OpenAI con Structured Outputs.
"""
import asyncio
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from prompts import (
    ToolDecision, 
    Response, 
    GoalEvaluation, 
    create_decision_template,
    create_evaluation_template,
    RESPONSE_TEMPLATE
)


# ==========================
# 📦 ESTADO DEL AGENTE
# ==========================

class AgentState(TypedDict):
    """Estado que se pasa entre nodos del grafo."""
    goal: str
    history: list[dict]
    current_step: str
    last_result: str
    completed: bool
    can_continue: bool
    iteration: int
    max_iterations: int
    final_answer: str
    tool_decision: dict  # Decisión de qué herramienta usar
    failed_attempts: dict  # Contador de intentos fallidos por herramienta


# ==========================
# 🧩 NODOS DEL GRAFO
# ==========================

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
        
        # ✅ CAMBIO: Usar method="function_calling" para mejor compatibilidad con OpenAI
        decision_template = create_decision_template(tools_description)
        self.decision_chain = decision_template | self.llm.with_structured_output(
            ToolDecision,
            method="function_calling"  # ← Evita errores de schema con OpenAI
        )
        
        evaluation_template = create_evaluation_template(tools_description)
        self.evaluation_chain = evaluation_template | self.llm.with_structured_output(
            GoalEvaluation,
            method="function_calling"  # ← Evita errores de schema con OpenAI
        )
    
    async def evaluate_node(self, state: AgentState) -> AgentState:
        """Evalúa el progreso y decide el siguiente paso."""
        print(f"\n--- Iteración {state['iteration']}/{state['max_iterations']} ---")
        print("📊 Evaluando progreso...")
        
        try:
            evaluation: GoalEvaluation = await self.evaluation_chain.ainvoke({
                "goal": state["goal"],
                "history": str(state["history"][-5:]),
                "last_result": state["last_result"]
            })
            
            print(f"🧠 Evaluación: {evaluation.reasoning}")
            print(f"📌 Estado: {evaluation.status}")
            
            # Mapear el status único a los campos del estado
            state["completed"] = evaluation.status == "completed"
            state["can_continue"] = evaluation.status == "in_progress"
            state["current_step"] = evaluation.next_step
            state["iteration"] += 1
            
        except Exception as e:
            print(f"❌ Error en evaluate_node: {str(e)}")
            state["can_continue"] = False
            state["completed"] = False
        
        return state
    
    async def decide_node(self, state: AgentState) -> AgentState:
        """Decide qué herramienta usar."""
        print(f"➡️  Siguiente paso: {state['current_step']}")
        
        try:
            decision: ToolDecision = await self.decision_chain.ainvoke({
                "question": state["current_step"]
            })
            
            print(f"🧠 Razonamiento: {decision.reasoning}")
            print(f"🔧 Tool sugerida: {decision.tool}")
            print(f"❓ Needs tool: {decision.needs_tool}")
            
            # ✅ NUEVO: Detectar loops infinitos
            if decision.tool and decision.needs_tool:
                failed_count = state.get("failed_attempts", {}).get(decision.tool, 0)
                if failed_count >= 3:
                    print(f"⚠️  LOOP DETECTADO: {decision.tool} ha fallado {failed_count} veces. Cancelando.")
                    state["can_continue"] = False
                    state["last_result"] = f"Loop detectado: {decision.tool} falló múltiples veces. Ya tengo suficiente información de la búsqueda inicial."
                    return state
            
            # Corrección: Si needs_tool=True pero tool está vacío, extraer del next_step
            if decision.needs_tool and not decision.tool:
                next_step_lower = state['current_step'].lower()
                
                # Usar las herramientas descubiertas dinámicamente
                # Ordenar por longitud descendente para evitar coincidencias parciales
                sorted_tools = sorted(self.available_tools, key=len, reverse=True)
                
                for tool_name in sorted_tools:
                    if tool_name in next_step_lower:
                        print(f"⚠️  Corrección: Detecté '{tool_name}' en el next_step.")
                        decision.tool = tool_name
                        
                        # Intentar extraer parámetros genéricamente del next_step
                        # Buscar patrones como: tool_name(param1=value1, param2=value2)
                        import re
                        params_match = re.search(rf'{tool_name}\s*\((.*?)\)', state['current_step'])
                        if params_match and not decision.arguments:
                            params_str = params_match.group(1)
                            # Parsear parámetros: ip=192.168.0.1, filename=test.txt
                            params = {}
                            for param in params_str.split(','):
                                if '=' in param:
                                    key, value = param.split('=', 1)
                                    params[key.strip()] = value.strip()
                            if params:
                                decision.arguments = params
                                print(f"⚠️  Corrección: Extraídos parámetros - {params}")
                        
                        break
            
            # Corrección: Si menciona una herramienta pero needs_tool=False, forzar
            if not decision.needs_tool and decision.tool:
                print(f"⚠️  Corrección: El LLM sugirió '{decision.tool}' pero marcó needs_tool=False. Forzando a True.")
                decision.needs_tool = True
            
            state["tool_decision"] = {
                "needs_tool": decision.needs_tool,
                "tool": decision.tool,
                "arguments": decision.arguments
            }
        except Exception as e:
            print(f"❌ Error en decide_node: {str(e)}")
            state["can_continue"] = False
            state["last_result"] = f"Error al decidir herramienta: {str(e)}"
        
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
        
        print(f"⚙️  Ejecutando: {tool_name}({args})")
        
        # Inicializar failed_attempts si no existe
        if "failed_attempts" not in state:
            state["failed_attempts"] = {}
        
        try:
            # Usar el cache de herramientas en lugar de llamar get_tools() cada vez
            tool_obj = self.tools_cache.get(tool_name)

            if not tool_obj:
                result_str = f"Error: Herramienta '{tool_name}' no encontrada"
                print(f"❌ {result_str}")
                # ✅ Incrementar contador de fallos
                state["failed_attempts"][tool_name] = state["failed_attempts"].get(tool_name, 0) + 1
            else:
                # Invocar la herramienta usando LangChain
                result = await tool_obj.ainvoke(args)
                result_str = str(result)[:3000]  
                print(f"✅ Resultado: {result_str}")
                # ✅ Reset contador si tuvo éxito
                state["failed_attempts"][tool_name] = 0
        except Exception as e:
            result_str = f"Error: {str(e)}"
            print(f"❌ {result_str}")
            # ✅ Incrementar contador de fallos
            state["failed_attempts"][tool_name] = state["failed_attempts"].get(tool_name, 0) + 1
        
        state["history"].append({
            "step": state["current_step"],
            "tool": tool_name,
            "args": args,
            "result": result_str
        })
        state["last_result"] = result_str
        
        return state
    
    async def finalize_node(self, state: AgentState) -> AgentState:
        """Genera la respuesta final."""
        print(f"\n{'='*60}")
        print("🎉 ¡OBJETIVO COMPLETADO!")
        print(f"{'='*60}\n")
        
        response: Response = await self.response_chain.ainvoke({
            "question": f"Resume lo que hiciste para cumplir: {state['goal']}",
            "tool_result": str(state["history"])
        })
        
        state["final_answer"] = response.answer
        
        return state


# ==========================
# 🔀 LÓGICA DE ROUTING
# ==========================

def should_continue(state: AgentState) -> Literal["decide", "finalize", "end"]:
    """Decide el siguiente nodo basándose en el estado."""
    
    # Si completó el objetivo → finalizar
    if state["completed"]:
        return "finalize"
    
    # Si no puede continuar o alcanzó límite → terminar
    if not state["can_continue"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    
    # Continuar con el siguiente paso
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
    
    return workflow.compile()


# ==========================
# 🎮 MAIN
# ==========================

async def main():
    # ✅ Configuración para OpenAI con gpt-4o-mini
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,
        timeout=120  # 2 minutos de timeout
    )
    
    client = MultiServerMCPClient({
        "shell": {
            "command": "python",
            "args": ["/Users/dani/Proyectos/mcp/shell_mcp_server_local.py"],
            "transport": "stdio",
        },
        "opensearch": {
            "command": "python",
            "args": ["/Users/dani/Proyectos/mcp/opensearch_mcp_server.py"],
            "transport": "stdio"
        }
    })
    
    # Crear el grafo (ahora es async)
    graph = await create_agent_graph(llm, client)
    
    print("="*60)
    print("🤖 AGENTE MCP con LangGraph (OpenAI)")
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
        
        # Estado inicial
        initial_state: AgentState = {
            "goal": user_input,
            "history": [],
            "current_step": "",
            "last_result": "Comenzando...",
            "completed": False,
            "can_continue": True,
            "iteration": 0,
            "max_iterations": 15,
            "final_answer": "",
            "tool_decision": {},
            "failed_attempts": {}  # ✅ Inicializar contador de fallos
        }
        
        # Ejecutar el grafo
        final_state = await graph.ainvoke(initial_state)
        
        if final_state.get("final_answer"):
            print(f"✅ {final_state['final_answer']}")
        else:
            print(f"⚠️  No se pudo completar el objetivo")


if __name__ == "__main__":
    asyncio.run(main())
