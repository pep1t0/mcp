"""
Ejemplo de agente autónomo con LLM y herramientas MCP.
El agente recibe un objetivo general, lo divide en pasos y los ejecuta usando herramientas MCP.
"""
import asyncio
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


# ==========================
# 📦 MODELOS PYDANTIC
# ==========================

class PlanModel(BaseModel):
    """Plan de acción dividido en pasos."""
    steps: List[str] = Field(description="Lista de pasos a ejecutar en orden")
    rationale: str = Field(description="Razonamiento general del plan")

class ToolDecision(BaseModel):
    """Decisión sobre qué herramienta usar."""
    tool: str = Field(description="Nombre de la herramienta a usar")
    arguments: dict = Field(description="Argumentos para la herramienta")
    rationale: str = Field(description="Por qué usar esta herramienta")

class GoalEvaluation(BaseModel):
    """Evaluación del cumplimiento del objetivo."""
    completed: bool = Field(description="Si el objetivo está completado")
    reasoning: str = Field(description="Razonamiento de la evaluación")
    next_action: Optional[str] = Field(default=None, description="Qué hacer si no está completo")

# ==========================
# 📋 TEMPLATES DE PROMPTS
# ==========================

PLANNER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Eres un experto en ciberseguridad que genera planes de acción detallados.
    
HERRAMIENTAS DISPONIBLES:
- scan_services(ip): Escanea puertos con nmap
- ftp_list_directory(ip, directory): Lista contenido FTP
- ftp_download_file(ip, filename, remote_directory, local_path): Descarga archivo FTP
- list_directory(path): Lista archivos locales
- execute_shell_command(command, working_dir): Ejecuta comando shell

Genera un plan paso a paso que sea CONCRETO y EJECUTABLE."""),
    ("human", "{goal}")
])

EXECUTOR_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Eres un agente autónomo de ciberseguridad que decide qué herramienta usar.

HERRAMIENTAS DISPONIBLES:
- scan_services(ip): Escanea puertos 21,22,80,443,990 con nmap -sV
- ftp_list_directory(ip, directory="/"): Lista contenido de directorio FTP anónimo
- ftp_download_file(ip, filename, remote_directory="/", local_path="/tmp"): Descarga archivo FTP
- list_directory(path): Lista archivos locales
- execute_shell_command(command, working_dir): Ejecuta comando shell

REGLAS:
1. Si ves "PORT 21/tcp open ftp" en resultados de nmap, usa ftp_list_directory
2. Si ves archivos en el listado FTP, descárgalos uno por uno con ftp_download_file
3. Extrae nombres de archivos exactos del output (última columna después de espacios)
4. Usa el historial para no repetir acciones"""),
    ("human", """Tarea actual: {step}

Historial de acciones:
{history}

Decide qué herramienta usar y con qué argumentos.""")
])

EVALUATOR_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Evalúa si el objetivo de seguridad se ha cumplido completamente.

CRITERIOS DE ÉXITO:
- Se escaneó la IP con nmap ✓
- Se identificaron servicios abiertos ✓
- Si FTP anónimo existe, se listaron archivos ✓
- Si hay archivos, se descargaron todos ✓"""),
    ("human", """Objetivo: {goal}

Último resultado: {context}

¿El objetivo está COMPLETAMENTE cumplido?""")
])


# ==========================
# 1️⃣ COMPONENTE: GoalManager
# ==========================

class GoalManager:
    def __init__(self, goal: str, llm):
        self.goal = goal
        self.completed = False
        self.llm = llm
        self.chain = EVALUATOR_TEMPLATE | llm.with_structured_output(GoalEvaluation)

    async def evaluate(self, context: str) -> bool:
        """Evalúa si el objetivo ya está cumplido."""
        evaluation: GoalEvaluation = await self.chain.ainvoke({
            "goal": self.goal,
            "context": context[:1000]  # Limitar contexto
        })
        
        print(f"📊 Evaluación: {evaluation.reasoning}")
        if not evaluation.completed and evaluation.next_action:
            print(f"➡️  Siguiente acción sugerida: {evaluation.next_action}")
        
        self.completed = evaluation.completed
        return self.completed


# =======================
# 2️⃣ COMPONENTE: Planner
# =======================

class Planner:
    def __init__(self, llm):
        self.llm = llm
        self.chain = PLANNER_TEMPLATE | llm.with_structured_output(PlanModel)

    async def generate_plan(self, goal: str) -> list[str]:
        """Divide el objetivo general en pasos concretos."""
        plan: PlanModel = await self.chain.ainvoke({"goal": goal})
        
        print(f"💡 Razonamiento del plan: {plan.rationale}\n")
        return plan.steps


# =======================
# 3️⃣ COMPONENTE: Executor
# =======================

class AgenticExecutor:
    def __init__(self, llm, client, goal_manager: GoalManager):
        self.llm = llm
        self.client = client
        self.goal_manager = goal_manager
        self.history = []
        self.chain = EXECUTOR_TEMPLATE | llm.with_structured_output(ToolDecision)

    async def execute_step(self, step: str):
        """El LLM decide qué tool usar para completar el paso."""
        decision: ToolDecision = await self.chain.ainvoke({
            "step": step,
            "history": json.dumps(self.history[-3:], indent=2)  # Solo últimas 3 acciones
        })
        
        print(f"\n🧩 Paso: {step}")
        print(f"🧠 Razonamiento: {decision.rationale}")
        print(f"⚙️  Ejecutando {decision.tool}({decision.arguments})\n")

        try:
            result = await self.client.call_tool(decision.tool, decision.arguments)
            result_str = str(result)[:500]  # Limitar tamaño para historial
        except Exception as e:
            result_str = f"❌ Error ejecutando herramienta: {str(e)}"
            print(result_str)
        
        self.history.append({
            "step": step,
            "tool": decision.tool,
            "args": decision.arguments,
            "result": result_str
        })
        
        return result


# =======================
# 4️⃣ MAIN: ciclo Agentic
# =======================

async def main():
    llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", temperature=0.2)

    client = MultiServerMCPClient({
        "shell": {
            "command": "python",
            "args": ["/Users/dani/Proyectos/mcp/shell_mcp_server_local.py"],
            "transport": "stdio",
        },
        "nmap": {
            "transport": "streamable_http",
            "url": "http://192.168.0.248:8080/mcp"
        }
    })

    # 🎯 NUEVO OBJETIVO: Análisis de seguridad y exfiltración de datos
    target_ip = "192.168.0.100"  # 🔧 CAMBIA ESTA IP A TU OBJETIVO
    
    goal_text = f"""
    Realiza un análisis de seguridad completo de la máquina {target_ip}:
    1. Escanea puertos comunes (21, 22, 80, 443, 990) con nmap
    2. Identifica servicios vulnerables (especialmente FTP con anonymous)
    3. Si encuentras FTP anónimo, lista todos los directorios
    4. Descarga TODOS los archivos encontrados a /tmp
    5. Genera un reporte final con los archivos descargados
    """
    
    goal_manager = GoalManager(goal_text, llm)
    planner = Planner(llm)
    executor = AgenticExecutor(llm, client, goal_manager)

    print(f"\n{'='*60}")
    print(f"🎯 OBJETIVO: Análisis de seguridad de {target_ip}")
    print(f"{'='*60}\n")
    
    plan = await planner.generate_plan(goal_text)
    print(f"🗺️  Plan generado ({len(plan)} pasos):")
    for i, step in enumerate(plan, 1):
        print(f"   {i}. {step}")
    print()

    MAX_ITERATIONS = 10  # Límite de seguridad
    iteration = 0
    
    for step in plan:
        iteration += 1
        if iteration > MAX_ITERATIONS:
            print(f"⚠️  Alcanzado límite de {MAX_ITERATIONS} iteraciones. Deteniendo.")
            break
            
        try:
            result = await executor.execute_step(step)
            print(f"✅ Resultado: {str(result)[:200]}...\n")
            
            done = await goal_manager.evaluate(str(result))
            if done:
                print(f"\n{'='*60}")
                print("🎉 ¡OBJETIVO ALCANZADO! Finalizando.")
                print(f"{'='*60}")
                break
            else:
                print("🔁 Continuando al siguiente paso...\n")
        except Exception as e:
            print(f"❌ Error en paso '{step}': {str(e)}")
            print("🔁 Continuando con el siguiente paso...\n")
    
    print(f"\n📋 Resumen de ejecución:")
    print(f"   - Total de pasos ejecutados: {iteration}")
    print(f"   - Objetivo completado: {'✅ Sí' if goal_manager.completed else '❌ No'}")
    print(f"   - Acciones realizadas: {len(executor.history)}")
    print()

if __name__ == "__main__":
    asyncio.run(main())
