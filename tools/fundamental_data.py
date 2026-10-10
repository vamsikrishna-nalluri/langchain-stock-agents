import yfinance as yf
from langchain_core.tools import tool

@tool
def get_stock_fundamentals(ticker: str) -> str:
    """
    Fetches fundamental financial metrics and the latest news headlines for a given stock ticker.
    Use this tool when the user asks about valuation, earnings, news, or fundamental analysis.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        news = stock.news
        
        # Extract key financial metrics safely
        pe_ratio = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        market_cap = info.get("marketCap", "N/A")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        summary = info.get("longBusinessSummary", "No summary available.")[:500] + "..."
        
        # Format the top 3-4 news headlines
        news_str = ""
        if news:
            for item in news[:4]:
                title = item.get("title", "No Title")
                publisher = item.get("publisher", "Unknown")
                news_str += f"- {title} ({publisher})\n"
        else:
            news_str = "No recent news found."
            
        report = (
            f"Company Profile: {sector} / {industry}\n"
            f"Business Summary: {summary}\n\n"
            f"Key Metrics:\n"
            f"- Market Cap: {market_cap}\n"
            f"- Trailing P/E: {pe_ratio}\n"
            f"- Forward P/E: {forward_pe}\n\n"
            f"Recent News Headlines:\n{news_str}"
        )
        return report
        
    except Exception as e:
        return f"An error occurred while fetching fundamental data for {ticker}: {str(e)}"