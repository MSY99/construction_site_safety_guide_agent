from datetime import datetime
from pytz import timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Get_Date")

@mcp.tool()
def get_requested_date():
    datetime.now(timezone('Asia/Seoul'))
    try:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y %m %d")

        return f"Current time in South Korea is: {formatted_time}"
    except Exception as e:
        return f"Error getting time: {str(e)}"
    
if __name__ == "__main__":
    mcp.run(transport="stdio")