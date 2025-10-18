"""
MCP + LangChain + Qwen3 - CON Agente ReAct
El agente decide automáticamente qué herramientas usar y las ejecuta
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

async def main():
    print("🤖 Sistema con Agente ReAct\n")
    
    # 1. Configurar servidor MCP
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
    
    # 2. Cargar herramientas
    tools = await client.get_tools()
    print(f"📦 Herramientas: {[t.name for t in tools]}\n")
    
    # 3. Configurar LLM
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0,
        timeout=60,
    )
    
    # 4. Crear agente (hace todo automático)
    agent = create_react_agent(llm, tools)
    
    # 5. Modo interactivo
    print("💬 Escribe tus comandos (o 'salir' para terminar)\n")
    print("Ejemplos:")
    print("  - Lista los archivos del directorio actual")
    print("  - ¿Cuántos archivos .py hay?")
    print("  - Ejecuta 'whoami' y dime quién soy")
    print("  - Muéstrame el uso del disco con df -h\n")
    print("  - ¿Hay alertas meteorológicas en California?")
    print("  - Dame el pronóstico para New York (latitud 40.7128, longitud -74.0060)\n")

    while True:
        try:
            user_input = input("🗣️  Tú: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break
            
            if not user_input:
                continue
            
            # El agente hace todo: decide qué herramientas usar y las ejecuta
            response = await agent.ainvoke({
                "messages": [HumanMessage(content=user_input)]
            })
            
            # Mostrar solo la respuesta final (sin el <think>)
            final_content = response["messages"][-1].content
            
            # Limpiar el output de <think> tags
            if "<think>" in final_content:
                final_content = final_content.split("</think>")[-1].strip()
            
            print(f"🤖 Agente: {final_content}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    print("👋 ¡Hasta luego!")

if __name__ == "__main__":
    asyncio.run(main())