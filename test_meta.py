from agent.services.arxiv_client import get_arxiv_client

client = get_arxiv_client()
p = client.fetch_metadata("2401.12345")
print(type(p).__name__)
print("url:", p.url)
print("pdf_url:", p.pdf_url)
