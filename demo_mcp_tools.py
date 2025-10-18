"""
Demo: Ver qué herramienta usa el agente para cada pregunta
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

async def demo_herramientas():
    print("🔍 DEMO: ¿Qué herramienta usa el agente?\n")
    
    # Configurar
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
    
    tools = await client.get_tools()
    print(f"📦 Herramientas disponibles:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    print()
    
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )
    
    agent = create_react_agent(llm, tools)
    
    # Diferentes preguntas
    queries = [
        "Lista los archivos del directorio actual",
        "Ejecuta el comando 'whoami' y dime quién soy",
        "Muestra el uso del disco con df -h",
        "¿Qué archivos hay en /tmp?",
    ]
    
    for query in queries:
        print(f"{'='*60}")
        print(f"🗣️  Pregunta: {query}")
        print(f"{'='*60}")
        
        response = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Analizar qué herramientas se usaron
        print("🔧 Herramientas usadas:")
        for msg in response["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"   ✅ {tool_call['name']}(")
                    for key, value in tool_call['args'].items():
                        print(f"      {key}={value}")
                    print(f"   )")
        
        # Respuesta final
        final_content = response["messages"][-1].content
        if "<think>" in final_content:
            final_content = final_content.split("</think>")[-1].strip()
        print(f"\n🤖 Respuesta: {final_content[:200]}...\n")

if __name__ == "__main__":
    asyncio.run(demo_herramientas())