import sys
import io
import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

agent = Agent(model=Groq(id="llama-3.3-70b-versatile"))

def log_message(message):
    """Write logs to stderr to prevent JSON parsing issues in Node.js"""
    sys.stderr.write(message + "\n")

def fetch_dynamic_html(url, chunk_size=1000):
    """Scrape, clean, and split HTML content from a dynamic webpage."""

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")

    driver = webdriver.Edge(options=options)

    log_message("Loading webpage...")  
    driver.get(url)
    time.sleep(5)

    html = driver.page_source
    driver.quit()

    log_message("Webpage loaded successfully!") 
    cleaned_text = preprocess_html(html)
    return split_into_chunks(cleaned_text, chunk_size)

def preprocess_html(html):
    """Clean HTML by extracting meaningful content."""
    soup = BeautifulSoup(html, "html.parser")

    remove_tags = ["script", "style", "meta", "noscript", "iframe", "link", "svg",
                   "footer", "header", "aside", "nav", "form", "button"]
    for tag in remove_tags:
        for element in soup.find_all(tag):
            element.extract()

    important_tags = ["h1", "h2", "h3", "h4", "p", "span", "li", "div", "td", "th"]
    content_set = set()

    for tag in important_tags:
        for element in soup.find_all(tag):
            text = element.get_text(strip=True).lower()
            if text and len(text) > 10 and not is_irrelevant_text(text):
                content_set.add(text)

    return "\n".join(sorted(content_set, key=len, reverse=True))

def is_irrelevant_text(text):
    """Filter out common repetitive text."""
    ignore_phrases = ["share this property", "more cities", "view more", "show map",
                      "advertise", "get alerts", "currency:", "copy link",
                      "facebook", "twitter", "whatsapp", "pinterest", "linkedin"]
    return any(phrase in text for phrase in ignore_phrases)

def split_into_chunks(text, chunk_size):
    """Split text into chunks."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def query_agent(chunks, query):
    """Query AI agent with extracted text and user query."""
    for i, chunk in enumerate(chunks):
        log_message(f"🔹 Processing Chunk {i+1}...")  

        prompt = f"""
        Extract the following detail from the text:
        - {query}
        
        {chunk}

answer should be in this format:
e.g. price: Rs.700 (just write what is asked)
for price please do include the currency
        """

        try:
            response = agent.run(prompt)
            if response and hasattr(response, 'content'):
                extracted_data = response.content.strip()
                if extracted_data:
                    return extracted_data
        except Exception as e:
            log_message(f"Error in AI response: {str(e)}")  

    return None

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing URL or Query parameter"}))
        return

    url = sys.argv[1]
    query = sys.argv[2]

    chunks = fetch_dynamic_html(url, chunk_size=1000)
    extracted_data = query_agent(chunks, query)

    result = {
        "url": url,
        "query": query,
        "extracted_data": extracted_data if extracted_data else "No relevant data found"
    }

    print(json.dumps(result)) 

if __name__ == "__main__":
    main()
