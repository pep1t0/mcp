"""
Demo con LOGGING DETALLADO del proceso de decisión del LLM
Muestra TODO el razonamiento interno del agente
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from datetime import datetime
import json

class MCPLogger:
    """Logger detallado para el proceso de decisión del agente"""
    
    def __init__(self):
        self.step = 0
    
    def log_section(self, title: str):
        """Imprime una sección destacada"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    def log_step(self, description: str):
        """Imprime un paso del proceso"""
        self.step += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] PASO {self.step}: {description}")
    
    def log_thinking(self, thinking: str):
        """Muestra el razonamiento interno del LLM"""
        print(f"\n💭 RAZONAMIENTO INTERNO DEL LLM:")
        print(f"{'─'*70}")
        # Limpiar y formatear el pensamiento
        thinking = thinking.replace("<think>", "").replace("</think>", "").strip()
        for line in thinking.split('\n'):
            if line.strip():
                print(f"   {line.strip()}")
        print(f"{'─'*70}\n")
    
    def log_tool_selection(self, tool_calls: list):
        """Muestra qué herramientas seleccionó el LLM"""
        print(f"🔧 HERRAMIENTAS SELECCIONADAS:")
        for i, tool_call in enumerate(tool_calls, 1):
            print(f"\n   Herramienta #{i}:")
            print(f"   ├─ Nombre: {tool_call['name']}")
            print(f"   ├─ ID: {tool_call['id']}")
            print(f"   └─ Argumentos:")
            for key, value in tool_call['args'].items():
                print(f"      • {key}: {value}")
    
    def log_tool_execution(self, tool_name: str, result: str):
        """Muestra el resultado de ejecutar una herramienta"""
        print(f"\n⚙️  EJECUCIÓN DE HERRAMIENTA: {tool_name}")
        print(f"{'─'*70}")
        # Truncar resultado si es muy largo
        if len(result) > 500:
            print(f"   {result[:500]}...")
            print(f"   [...truncado, total {len(result)} caracteres]")
        else:
            print(f"   {result}")
        print(f"{'─'*70}\n")
    
    def log_final_response(self, response: str):
        """Muestra la respuesta final al usuario"""
        print(f"✅ RESPUESTA FINAL AL USUARIO:")
        print(f"{'─'*70}")
        print(f"{response}")
        print(f"{'─'*70}\n")

async def demo_con_logging():
    logger = MCPLogger()
    
    logger.log_section("🔍 DEMO: Decisiones del LLM en Tiempo Real")
    
    # Configurar servidor MCP
    logger.log_step("Inicializando servidor MCP")
    client = MultiServerMCPClient({
        "shell": {
            "command": "python",
            "args": ["/Users/dani/Projectes/mcp/shell_mcp_server_local.py"],
            "transport": "stdio",
        },
        "weather": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",  # URL del servidor weather_mcp_server_remote.py
        }
    })
    
    logger.log_step("Cargando herramientas disponibles")
    tools = await client.get_tools()
    
    print(f"📦 Herramientas cargadas:")
    for tool in tools:
        print(f"   • {tool.name}")
        desc_lines = tool.description.split('\n')
        print(f"     └─ {desc_lines[0][:60]}...")
    print()
    
    # Configurar LLM
    logger.log_step("Inicializando modelo Qwen3")
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )
    
    # Crear agente
    logger.log_step("Creando agente ReAct")
    agent = create_react_agent(llm, tools)
    
    # Preguntas de prueba
    queries = [
        "Lista los archivos del directorio actual",
        "Ejecuta el comando 'whoami' y dime quién soy",
        "¿Cuántos archivos .py hay en el directorio actual?",
        "¿Hay alertas meteorológicas en California?",
        "Dame el pronóstico para New York (latitud 40.7128, longitud -74.0060)"
    ]
    
    for query_num, query in enumerate(queries, 1):
        logger.log_section(f"QUERY #{query_num}: {query}")
        
        logger.log_step(f"Usuario pregunta: '{query}'")
        
        # Invocar agente
        logger.log_step("Invocando agente...")
        response = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Analizar el flujo de mensajes
        messages = response["messages"]
        
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                logger.log_step(f"Mensaje del usuario procesado")
                
            elif isinstance(msg, AIMessage):
                # Primera respuesta del LLM
                if msg.tool_calls:
                    logger.log_step("LLM decide usar herramientas")
                    
                    # Extraer y mostrar el razonamiento
                    if "<think>" in msg.content:
                        thinking = msg.content.split("<think>")[1].split("</think>")[0]
                        logger.log_thinking(thinking)
                    
                    # Mostrar herramientas seleccionadas
                    logger.log_tool_selection(msg.tool_calls)
                    
                else:
                    # Respuesta final
                    logger.log_step("LLM genera respuesta final")
                    
                    # Mostrar razonamiento final si existe
                    if "<think>" in msg.content:
                        thinking = msg.content.split("<think>")[1].split("</think>")[0]
                        logger.log_thinking(thinking)
                    
                    # Extraer respuesta limpia
                    response_text = msg.content
                    if "</think>" in response_text:
                        response_text = response_text.split("</think>")[1].strip()
                    
                    logger.log_final_response(response_text)
                    
            elif isinstance(msg, ToolMessage):
                # Resultado de ejecutar herramienta
                logger.log_step("Resultado de herramienta recibido")
                # El nombre de la herramienta está en el mensaje anterior (AIMessage)
                logger.log_tool_execution("herramienta", msg.content)
        
        print("\n" + "🎯" * 35 + "\n")
        
        # Pausa para legibilidad
        await asyncio.sleep(0.5)

async def demo_interactivo_con_logging():
    """Versión interactiva con logging"""
    logger = MCPLogger()
    
    logger.log_section("💬 Modo Interactivo con Logging Detallado")
    
    # Configurar servidor MCP
    logger.log_step("Inicializando servidor MCP")
    client = MultiServerMCPClient({
        "shell": {
            "command": "python",
            "args": ["/Users/dani/Projectes/mcp/shell_mcp_server.py"],
            "transport": "stdio",
        }
    })
    
    logger.log_step("Cargando herramientas disponibles")
    tools = await client.get_tools()
    
    print(f"📦 Herramientas cargadas:")
    for tool in tools:
        print(f"   • {tool.name}")
        desc_lines = tool.description.split('\n')
        print(f"     └─ {desc_lines[0][:60]}...")
    print()
    
    # Configurar LLM (añadido)
    logger.log_step("Inicializando modelo Qwen3")
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    # Crear agente (añadido)
    logger.log_step("Creando agente ReAct")
    agent = create_react_agent(llm, tools)
    
    print("💡 Escribe tus preguntas (o 'salir' para terminar)")
    print("💡 Verás TODO el proceso de razonamiento del LLM\n")
    
    while True:
        try:
            user_input = input("🗣️  Tú: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break
            
            if not user_input:
                continue
            
            logger.step = 0  # Reset step counter
            logger.log_section(f"QUERY: {user_input}")
            
            logger.log_step(f"Usuario pregunta: '{user_input}'")
            
            # Invocar agente
            response = await agent.ainvoke({
                "messages": [HumanMessage(content=user_input)]
            })
            
            # Analizar respuesta
            messages = response["messages"]
            
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        # Razonamiento
                        if "<think>" in msg.content:
                            thinking = msg.content.split("<think>")[1].split("</think>")[0]
                            logger.log_thinking(thinking)
                        
                        # Herramientas
                        logger.log_tool_selection(msg.tool_calls)
                        
                    else:
                        # Respuesta final
                        logger.log_step("LLM genera respuesta final")

                        # Mostrar razonamiento final si existe
                        if "<think>" in msg.content:
                            thinking = msg.content.split("<think>")[1].split("</think>")[0]
                            logger.log_thinking(thinking)
                        
                        # Extraer respuesta limpia
                        response_text = msg.content
                        if "</think>" in response_text:
                            response_text = response_text.split("</think>")[1].strip()
                        
                        logger.log_final_response(response_text)
                
                elif isinstance(msg, ToolMessage):
                    logger.log_tool_execution("herramienta", msg.content[:200] + "...")
            
            print("\n" + "─" * 70 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    print("\n👋 ¡Hasta luego!")

async def main():
    # Menú
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║         🔍 SISTEMA DE LOGGING DETALLADO - MCP + LLM           ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    print("Elige una opción:")
    print("  1. Demo automática (con ejemplos predefinidos)")
    print("  2. Modo interactivo (escribe tus propias preguntas)")
    print()
    
    choice = input("Opción (1/2): ").strip()
    
    if choice == "1":
        await demo_con_logging()
    elif choice == "2":
        await demo_interactivo_con_logging()
    else:
        print("Opción no válida")

if __name__ == "__main__":
    asyncio.run(main())