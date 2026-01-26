import asyncio
from mcp.server import FastMCP
import wikipedia

mcp = FastMCP("wikipedia")

# Health check
@mcp.tool()
async def ping() -> str:
    return "Wikipedia MCP is running!"

# Search Wikipedia
@mcp.tool()
async def search(query: str, limit: int = 5):
    results = wikipedia.search(query, results=limit)
    data = []
    for r in results:
        try:
            summary = wikipedia.summary(r, sentences=2, auto_suggest=False)
        except Exception:
            summary = "Summary not available"
        data.append({
            "title": r,
            "summary": summary,
            "url": f"https://en.wikipedia.org/wiki/{r.replace(' ', '_')}"
        })
    return data

# Get a page summary
@mcp.tool()
async def get_page(title: str):
    try:
        summary = wikipedia.summary(title, auto_suggest=False)
        return {
            "title": title,
            "summary": summary,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
