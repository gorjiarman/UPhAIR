import requests
import json
import os
from urllib.parse import urlencode
import config
import requests
import xml.etree.ElementTree as ET
import time
import re

def load_json_store(filename=config.PAPER_DIR+"/papers.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

class MultiSourcePaperFetcher:
    def __init__(self, unpaywall_email):
        self.unpaywall_email = unpaywall_email
        self.nume = 0

    def search_crossref(self, query, limit=20):
        url = "https://api.crossref.org/works"
        params = {"query": query, "rows": limit}
        resp = requests.get(url, params=params)
        data = resp.json()

        results = []
        for item in data.get("message", {}).get("items", []):
            results.append({
                "source": "CrossRef",
                "title": item.get("title", [""])[0],
                "uid": self.nume,
                "doi": item.get("DOI"),
                "pdf_url": None,  # Crossref doesn't provide PDFs
            })
            self.nume=self.nume+1
        return results

    def search_pubmed(self, query, limit=20):
        # Step 1: Search PubMed
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json"
        }
        search_resp = requests.get(search_url, params=params).json()
        ids = search_resp.get("esearchresult", {}).get("idlist", [])

        if not ids:
            return []

        # Step 2: Fetch metadata
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml"
        }
        xml_text = requests.get(fetch_url, params=fetch_params).text

        root = ET.fromstring(xml_text)

        # Map PMID → DOI
        pmid_to_doi = {}

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            doi = None

            for article_id in article.findall(".//ArticleId"):
                if article_id.attrib.get("IdType") == "doi":
                    doi = article_id.text
                    break

            pmid_to_doi[pmid] = doi

        # Step 3: Build results
        results = []
        for pmid in ids:
            print(pmid_to_doi.get(pmid))
            results.append({
                "source": "PubMed",
                "title": f"PubMed Article {pmid}",  # still placeholder
                "uid": self.nume,
                "doi": pmid_to_doi.get(pmid),
                "pdf_url": None
            })
            self.nume += 1

        return results

    def save_results_to_json(self, results, filename=config.PAPER_DIR+"/papers.json"):
        existing = load_json_store(filename)

        for item in results:
            record = {
                "source": item.get("source"),
                "title": item.get("title"),
                "doi": item.get("doi"),
                "pdf_url": item.get("pdf_url"),
                "uid": item.get("uid")
            }

            # Avoid duplicates (prefer DOI, fallback to title)
            if not any(
                (r.get("doi") and r.get("doi") == record["doi"]) or
                (r.get("title") == record["title"])
                for r in existing
            ):
                existing.append(record)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def get_unpaywall_pdf(self, doi):
        if not doi:
            return None
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": self.unpaywall_email}
            resp = requests.get(url, params=params).json()

            oa = resp.get("best_oa_location")
            if oa and oa.get("url_for_pdf"):
                return oa["url_for_pdf"]
        except:
            return None

    def download_pdf(self, url, save_path):
        if not url:
            return False

        start_time = time.time()
        timeout_seconds = 10

        try:
            with requests.get(url, stream=True, allow_redirects=True, timeout=(5, 5)) as resp:

                if not resp.ok:
                    return False

                content_type = resp.headers.get("Content-Type", "").lower()
                if "pdf" not in content_type:
                    return False

                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                        # HARD time limit
                        if time.time() - start_time > timeout_seconds:
                            return False
                print("✅ paper downloaded ✅")
                return True

        except requests.RequestException:
            return False


    def fetch(self, query, max_results=10, save_dir=config.PAPER_DIR):
        os.makedirs(save_dir, exist_ok=True)

        results = []
        print(f"Searching for: {query}")

        # Search three databases
        all_results = []
        all_results += self.search_crossref(query, limit=max_results)
        print("CrossRef search done."+str(len(all_results)))
        all_results += self.search_pubmed(query, limit=max_results)
        print("PubMed search done."+str(len(all_results))) 
        self.save_results_to_json(all_results)
        for idx, art in enumerate(all_results):
            doi = art["doi"]

            if art.get("title"):

                print(f"Processing article {idx+1}/{len(all_results)}: {art['title']}")
                # If no PDF from source, pull from Unpaywall
                if not art.get("pdf_url"):
                    art["pdf_url"] = self.get_unpaywall_pdf(doi)
                # Save PDF
                if art["pdf_url"]:
                    filename = str(art["uid"])+"--paper.pdf"
                    path = os.path.join(save_dir, filename)
                    if self.download_pdf(art["pdf_url"], path):
                        art["pdf_path"] = path

                results.append(art)

        return results
