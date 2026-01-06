import os
import json
import fitz  # PyMuPDF
import faiss
import re
from sentence_transformers import SentenceTransformer
import config
import ast
from scista import ArticleFetcher
from paper_downloader import load_json_store
from collections import defaultdict

class PaperRetriever:

    def fetch_paper_metadata(self):
        fetcher = ArticleFetcher(
            core_api_key=config.CORE_API_KEY,
            email_for_unpaywall=config.UNPAYWALL_EMAIL
        )

        articles = fetcher.fetch_articles(
            topic="glioma radiomics",
            num_articles=50,
            sort_by_date=True
        )

        # Open file once, write incrementally
        with open("data/paper_titles_and_dois.txt", "w", encoding="utf-8") as f:
            for idx, article in enumerate(articles, start=1):
                title = article.title or "N/A"
                doi = article.doi or "N/A"

                print("Title:", title)
                print("DOI:", doi)
                print("Text:", article.text)
                print("PDF URL:", article.pdf_url)

                # Write to file
                f.write(f"{idx}. Title: {title}\n")
                f.write(f"   DOI: {doi}\n\n")

                # Save PDF if available
                if article.pdf_url:
                    pdf_name = f"paper_{idx}.pdf"
                    success = article.save_pdf(pdf_name)
                    print("PDF saved?", success)
        
    
    def extract_text_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            return full_text
        except:
            return " not find "

    def remove_references(self, text):
        # Look for common reference section headers
        patterns = [
            r'\n\s*references\s*\n',
            r'\n\s*reference\s*\n',    # "References" on its own line
            r'\n\s*bibliography\s*\n',     # Some papers use "Bibliography"
            r'\n\s*literature cited\s*\n',
            r'\n\s*acknowledgments\s*\n'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return text[:match.start()]

        return text  # return full text if no pattern matches

    def extract_text_and_merge(self):
        paper_records = load_json_store(filename=config.PAPER_DIR+"/papers.json")
        all_texts = {}
        for paper_record in paper_records:
            fpath = config.PAPER_DIR+"/"+str(paper_record.get("uid"))+"--paper.pdf"
            print(fpath)
            pdf_text = self.extract_text_from_pdf(fpath)
            cleaned_text = self.remove_references(pdf_text)
            all_texts[paper_record.get("title")] = cleaned_text

            with open(config.DATA_DIR+"/new_dictionary.txt", "w") as f:
                f.write(str(all_texts))
            
class FaissRetriever:
    """
    Builds a FAISS vector store from text and retrieves relevant context.
    """

    def __init__(self, model_name=config.EMBEDDING_MODEL_NAME):
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.doc_ids = []

    def build_index_from_texts(self, feature_name, paper_texts, chunk_size=300, overlap=50):
        """
        Build FAISS index from feature-related paper chunks only
        """

        print("\nBuilding FAISS vector store from feature-related chunks...")

        self.documents = []
        self.doc_ids = []

        # ---- clean and tokenize feature ----
        base_feature = (
            feature_name.split(":")[0]
            .replace("*", "")
            .replace("-", "")
            .replace("_", " ")
            .strip()
            .lower()
        )

        feature_tokens = base_feature.split()

        for paper_id, text in paper_texts.items():
            if not text:
                continue

            words = text.split()
            start = 0

            while start < len(words):
                end = start + chunk_size
                chunk = " ".join(words[start:end])
                chunk_lower = chunk.lower()

                # ---- HARD LEXICAL FILTER (token-based) ----
                if not any(token in chunk_lower for token in feature_tokens):
                    start = end - overlap
                    continue

                self.documents.append(chunk)
                self.doc_ids.append(paper_id)

                start = end - overlap

        if not self.documents:
            print("No chunks matched the feature tokens.")
            return

        print(f"Embedding {len(self.documents)} filtered chunks...")

        embeddings = self.model.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))

        print(f"FAISS index built with {self.index.ntotal} chunk vectors.")

    def get_title_by_uid(self, uid, filename=config.PAPER_DIR + "/papers.json"):
        with open(filename, "r", encoding="utf-8") as f:
            papers = json.load(f)

        for paper in papers:
            if paper.get("uid") == int(uid):
                return paper.get("title")

    def retrieve_context(self, features, paper_texts, k=50):
        if self.index is None:
            return "Retriever index has not been built."

        print("\nQuerying FAISS index for relevant literature context...")

        base_feature = (
            features.split(":")[0]
            .replace("*", "")
            .replace("-", "")
            .replace("_", " ")
            .strip()
        )

        query_text = (
            f"Feature {base_feature} "
            f"associated with glioma grading, tumor heterogeneity, and prognosis"
        )

        print("Query text:", query_text)

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            k
        )

        # ---- aggregate retrieved chunks per paper ----
        paper_chunk_counts = defaultdict(int)

        for idx in indices[0]:
            paper_id = self.doc_ids[idx]
            paper_chunk_counts[paper_id] += 1

        if not paper_chunk_counts:
            return None, ""

        # ---- choose paper with most supporting chunks ----
        best_paper = max(
            paper_chunk_counts.items(),
            key=lambda x: x[1]
        )[0]

        # ---- retrieve FULL paper text ----
        full_text = paper_texts.get(best_paper, "")

        if not full_text:
            print("Warning: full text not found for selected paper.")
            return best_paper, ""
        print(
            f"📑 Selected paper {best_paper} "
            f"with {paper_chunk_counts[best_paper]} supporting chunks."
        )

        return best_paper, full_text
# --- Helper function to simulate txt extraction ---
# In a real scenario, this would involve downloading and parsing hundreds of PDFs,
# which is slow and error-prone. We will simulate by creating a dummy text file.
def get_paper_texts(force_create=False):
    """
    Simulates the process of extracting text from all papers.
    If a cached file exists, it loads it. Otherwise, it creates a dummy file.
    """
    if os.path.exists(config.ALL_TEXTS_PATH) and not force_create:
        print(f"Loading cached paper texts from {config.ALL_TEXTS_PATH}...")
        with open(config.ALL_TEXTS_PATH, "r") as f:
            return ast.literal_eval(f.read())
    else:
        print("Simulating paper download and text extraction...")
        # This is a placeholder. In a real workflow, you would download PDFs
        # from the DOIs and extract their text.
        dummy_texts = {
            "Paper_on_Age_and_Glioma": "Age is a significant prognostic factor in glioma. Older patients, typically over 55, are more likely to have IDH-wildtype glioblastoma, which is associated with a poorer prognosis. In contrast, IDH-mutant gliomas are more common in younger patients.",
            "Paper_on_MGMT_in_Glioblastoma": "O6-methylguanine-DNA methyltransferase (MGMT) promoter methylation is a key biomarker in glioblastoma. Methylation silences the gene, leading to better response to temozolomide chemotherapy. MGMT methylation is more frequently observed in IDH-mutant gliomas.",
            "Paper_on_Radiomics_Sphericity": "Radiomic features quantify tumor characteristics from medical images. Tumor sphericity, a measure of how spherical a tumor is, has been linked to prognosis. Lower sphericity, indicating a more irregular shape, is often associated with more aggressive, infiltrative tumors like IDH-wildtype glioblastoma.",
            "Paper_on_GLCM_Features": "Gray-Level Co-occurrence Matrix (GLCM) features describe tumor texture. High cluster prominence can indicate heterogeneity and structural complexity within the tumor, which may be characteristic of high-grade gliomas."
        }
        with open(config.ALL_TEXTS_PATH, 'w') as f:
            json.dump(dummy_texts, f)
        print(f"Saved dummy paper texts to {config.ALL_TEXTS_PATH}")
        return dummy_texts