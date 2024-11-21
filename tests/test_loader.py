import requests
from bs4 import BeautifulSoup

urls = [
    "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html",
    "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html"
]

def fetch_content(url):
    try:
        print(f"Fetching URL: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract main content (modify the tag selection as per the site structure)
        paragraphs = soup.find_all("p")
        content = "\n".join([para.get_text() for para in paragraphs])
        
        return content if content else "No content found."
    except Exception as e:
        return f"Error fetching content: {e}"

for url in urls:
    content = fetch_content(url)
    print(f"\n--- Content from {url} ---\n{content}\n")
