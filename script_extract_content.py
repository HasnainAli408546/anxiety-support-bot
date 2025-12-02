import requests
from bs4 import BeautifulSoup
import os
import re
import json

# import pdfplumber  # Commented out for now


def download_pdf(url, save_path):
    print(f"Downloading PDF from {url}")
    r = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {save_path}")


def extract_pdf_text(pdf_path):
    print(f"Extracting text from {pdf_path}")
    text = ""
    # with pdfplumber.open(pdf_path) as pdf:
    #     for page in pdf.pages:
    #         text += page.extract_text() + "\n"
    return text


def chunk_text(text, min_len=150, max_len=700):
    """Chunk by paragraph, heading, or manual delimiters."""
    if not text or len(text) < min_len:
        return []
    
    # Split by double newline or bullet points
    raw_chunks = re.split(r'(\n{2,}|•|\*|\- )', text)
    # Clean and join small fragments
    chunks = []
    buf = ""
    for chunk in raw_chunks:
        c = chunk.strip()
        if not c or len(c) < min_len:
            buf += f" {c}"
        else:
            chunks.append((buf + " " + c).strip())
            buf = ""
    if buf:
        chunks.append(buf.strip())
    # Filter out extra small/empty
    return [c for c in chunks if len(c) > min_len and len(c) < max_len]


def scrape_and_chunk_webpage(url, main_selector, scenario, content_type):
    """Scrape a website (e.g. NHS, Anxiety Canada) and chunk main content."""
    print(f"Scraping: {url}")
    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Select main text block(s) by CSS selector
        blocks = soup.select(main_selector)
        
        if not blocks:
            print(f"⚠️ No content found with selector '{main_selector}' for {url}")
            print("Available main elements:")
            for tag in ['main', 'article', '.content', '.main-content', 'div.content']:
                if soup.select(tag):
                    print(f"  - {tag}")
            return []
        
        # Grab text and split into paragraphs
        blocks = [b.get_text(separator="\n") for b in blocks]
        text = "\n\n".join(blocks)
        
        print(f"✅ Scraped {len(text)} characters from {url}")
        
        chunks = chunk_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        return [
            {
                "content": chunk.strip(),
                "metadata": {
                    "scenario": scenario,
                    "type": content_type,
                    "source": url
                }
            }
            for chunk in chunks
        ]
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error scraping {url}: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error scraping {url}: {e}")
        return []


def process_manual_pdf(pdf_url, scenario, content_type, output_dir="manuals/"):
    # PDF processing is commented out for now
    print(f"PDF processing is currently disabled for {pdf_url}")
    return []
    
    # os.makedirs(output_dir, exist_ok=True)
    # filename = pdf_url.split("/")[-1]
    # save_path = os.path.join(output_dir, filename)
    # download_pdf(pdf_url, save_path)
    # raw_text = extract_pdf_text(save_path)
    # chunks = chunk_text(raw_text)
    # return [
    #     {
    #         "content": chunk.strip(),
    #         "metadata": {
    #             "scenario": scenario,
    #             "type": content_type,
    #             "source": pdf_url
    #         }
    #     }
    #     for chunk in chunks
    # ]


def main():
    print("🚀 Starting content extraction (PDF disabled)")
    print("=" * 50)
    
    all_chunks = []
    
    # Example 1: PDF processing (commented out)
    # pdf_url = "https://web.ntw.nhs.uk/selfhelp/leaflets/Anxiety.pdf"
    # panic_chunks = process_manual_pdf(pdf_url, scenario="panic", content_type="education")
    panic_chunks = []  # Empty list since PDF is disabled
    print(f"PDF chunks: {len(panic_chunks)} (PDF processing disabled)")

    # Example 2: Scrape NHS Anxiety Self-Help page
    print("\n🌐 Scraping NHS content...")
    web_url = "https://www.nhs.uk/mental-health/self-help/guides-tools-and-activities/anxiety-tips/"
    page_chunks = scrape_and_chunk_webpage(
        url=web_url,
        main_selector="main",  # Simplified selector
        scenario="panic",
        content_type="technique"
    )
    all_chunks.extend(page_chunks)

    # Example 3: Scrape Anxiety Canada cognitive therapy guide
    print("\n🌐 Scraping Anxiety Canada content...")
    canada_url = "https://www.anxietycanada.com/articles/how-to-do-cbt/"
    cbt_chunks = scrape_and_chunk_webpage(
        canada_url, 
        "div.article-content, .content, main", 
        "uncertainty", 
        "education"
    )
    all_chunks.extend(cbt_chunks)
    
    # Example 4: Additional NHS content for more scenarios
    print("\n🌐 Scraping additional NHS content...")
    sleep_url = "https://www.nhs.uk/every-mind-matters/mental-health-issues/sleep/"
    sleep_chunks = scrape_and_chunk_webpage(
        sleep_url,
        "main",
        "sleep",
        "technique"
    )
    all_chunks.extend(sleep_chunks)
    
    # Example 5: Mind.org content
    print("\n🌐 Scraping Mind.org content...")
    mind_url = "https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/self-care/"
    mind_chunks = scrape_and_chunk_webpage(
        mind_url,
        "main, .content",
        "panic",
        "education"
    )
    all_chunks.extend(mind_chunks)

    # Combine all chunks (panic_chunks is now defined as empty list)
    total_chunks = panic_chunks + page_chunks + cbt_chunks + sleep_chunks + mind_chunks
    all_chunks = total_chunks  # Use total_chunks for consistency

    # Save to JSONL for easy population
    output_file = "rag_content_collected.jsonl"
    if all_chunks:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in all_chunks:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n✅ Collected & chunked {len(all_chunks)} items.")
        print(f"✅ Saved to {output_file}")
        
        # Show breakdown by scenario
        scenarios = {}
        for chunk in all_chunks:
            scenario = chunk['metadata']['scenario']
            scenarios[scenario] = scenarios.get(scenario, 0) + 1
        
        print(f"\n📊 Content breakdown by scenario:")
        for scenario, count in scenarios.items():
            print(f"  {scenario}: {count} chunks")
        
        print("\nFirst example chunk:")
        if all_chunks:
            first_chunk = all_chunks[0]
            print(f"Scenario: {first_chunk['metadata']['scenario']}")
            print(f"Type: {first_chunk['metadata']['type']}")
            print(f"Content: {first_chunk['content'][:200]}...")
    else:
        print("❌ No content was successfully extracted")

    print(f"\n🎉 Extraction complete!")


if __name__ == "__main__":
    main()
