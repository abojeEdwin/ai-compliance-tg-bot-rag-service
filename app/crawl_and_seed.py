import os
import requests
from dotenv import load_dotenv
from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions

load_dotenv()

firecrawl_app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
INGEST_URL = "http://localhost:8000/api/ingest"

def split_into_parent_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def split_into_child_chunks(parent_chunk: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(parent_chunk):
        end = start + chunk_size
        chunks.append(parent_chunk[start:end])
        start += chunk_size - overlap
    return chunks

def process_scraped_page(url: str, title: str, markdown_content: str, category: str):
    """Processes a single scraped page through the parent-child vector pipeline."""
    if not markdown_content or len(markdown_content.strip()) < 50:
        return

    print(f"📄 Slicing Page: {title or url} ({category})")
    parent_blocks = split_into_parent_chunks(markdown_content)
    
    success_count = 0
    for idx, parent_text in enumerate(parent_blocks):
        child_snippets = split_into_child_chunks(parent_text)
        
        payload = {
            "parent_text": parent_text,
            "child_chunks": child_snippets,
            "metadata": {
                "source_url": url,
                "title": title,
                "type": category,
                "block_index": idx
            }
        }
        
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = requests.post(INGEST_URL, json=payload)
                if res.status_code == 200:
                    success_count += 1
                    break
                elif res.status_code == 500 and "429" in res.text:
                    print(f"   ⚠️ Rate limited. Waiting 10 seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(10)
                else:
                    print(f"   ❌ DB pipeline returned status {res.status_code}: {res.text}")
                    break
            except Exception as e:
                print(f"   ❌ DB pipeline connection error: {e}")
                break
        
        # Base delay to prevent hitting the 15 RPM limit on the free tier
        time.sleep(2)
            
    print(f"   💾 Ingested {success_count}/{len(parent_blocks)} blocks into database.\n")

def run_knowledge_crawl():
    targets = [
        {
            "url": "https://doc.youverify.co",
            "category": "developer_api",
            "limit": 40, # Limit page counts so you don't burn all your free tokens on one crawl
            "paths": ["/get-started*", "/guides-and-solutions*", "/api-reference/api-reference*", "/api-reference/api-reference/activity-management*", "/our-legacy-api-and-sdk"]
        },
        #{
            #"url": "https://youverify.co",
            #"category": "commercial_business",
            #"limit": 15,
            #"paths": ["/en/resources/faqs*", "/en/company/about-us*", "/en/resources/regional-compliance*", "/en/blogs*", "/en/resources/country-coverage*", "/en/solution/customer-onboarding*", "/en/company/press-and-media"]
        #}
    ]

    for target in targets:
        print(f"🕸️ Spawning Firecrawl automation for: {target['url']}...")
        
        try:
            # Executes a synchronous crawl pattern. 
            # Firecrawl polls the background job status automatically and returns when ready.
            crawl_result = firecrawl_app.crawl(
                url=target["url"],
                limit=target["limit"],
                include_paths=target["paths"],
                scrape_options=ScrapeOptions(formats=["markdown"])
            )
            
            print(f"✅ Discovered & processed {len(crawl_result.data)} pages.")
            
            # Pipe results down into your DB ingestion flow
            for page in crawl_result.data:
                url = page.metadata.source_url if page.metadata else target["url"]
                title = page.metadata.title if page.metadata else "Untitled Section"
                markdown = page.markdown
                
                process_scraped_page(url, title, markdown, target["category"])
                
        except Exception as e:
            print(f"❌ Firecrawl crawl routine failed: {e}")

if __name__ == "__main__":
    print("🚀 Booting Crawl & DB Generation Routine...\n")
    run_knowledge_crawl()