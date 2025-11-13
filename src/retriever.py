import os
import json
import requests
import fitz  # PyMuPDF
import faiss
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import config
import glob
import ast
class PaperRetriever:
    """
    Handles fetching paper metadata from Europe PMC.
    (Note: Downloading and parsing PDFs is a heavy task and is simplified here)
    """
    def fetch_paper_metadata(self):
        print("\nFetching paper metadata from Europe PMC...")
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": config.PUBMED_QUERY,
            "format": "json",
            "pageSize": config.MAX_PUBMED_RESULTS
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("resultList", {}).get("result", [])
            
            papers = {}
            for paper in results:
                title = paper.get("title", "").strip()
                # A simple way to create a filename-safe title
                filename = "".join([c for c in title if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
                if title:
                    papers[filename] = paper.get("doi", "")
            
            print(f"Found metadata for {len(papers)} papers.")
            return papers
        except requests.RequestException as e:
            print(f"❌ Error fetching from Europe PMC: {e}")
            return {}
    
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

    def download_papers(self, metadatas):
        """
        Downloads the PDF from the DOI link and extracts text.
        Note: This is a simplified placeholder function.
        """
        for metadata in metadatas.items():
            title, doi = metadata
            print(f"\nProcessing paper: {title}")
            print(f"Downloading and extracting text for DOI: {doi}")
            current_files = glob.glob(config.PAPER_DIR)
            if f"{title}.pdf" in current_files:
                continue

            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={config.UNPAYWALL_EMAIL}"
            try:
                # Step 1: Query Unpaywall for OA PDF
                res = requests.get(unpaywall_url).json()
                if res.get("is_oa") and res.get("oa_locations"):
                    pdf_url = res["oa_locations"][0].get("url_for_pdf")
                    if pdf_url:
                        # Step 2: Download the PDF
                        pdf_response = requests.get(pdf_url)
                        if pdf_response.status_code == 200:
                            # Optional: Create output folder
                            filename = title+".pdf"
                            filepath = os.path.join(config.PAPER_DIR, filename)

                            with open(filepath, "wb") as f:
                                f.write(pdf_response.content)

                            print(f"✅ PDF downloaded and saved to: {filepath}")
                        else:
                            print("⚠️ Failed to download the PDF (bad response).")
                    else:
                        print("⚠️ No direct PDF URL found in the OA record.")
                else:
                    print("❌ No open access PDF found for this DOI.")
            except:
                print("No DOI")
    
    def extract_text_and_merge(self):

        all_papers = glob.glob(config.PAPER_DIR+"/*.pdf")
        # print(all_papers)
        # Extract text and analyze
        all_texts = {}
        for paper in all_papers:
            print(paper)
            pdf_text = self.extract_text_from_pdf(paper)
            cleaned_text = self.remove_references(pdf_text)
            all_texts[paper.split("/")[-1].replace(".pdf","")] = cleaned_text

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

    def build_index_from_texts(self, paper_texts):
        """
        Embeds documents and builds a FAISS index.
        """
        print("\nBuilding FAISS vector store from paper texts...")
        
        for name, content in paper_texts.items():

            self.documents.append(content)
            self.doc_ids.append(name)
        
        print(f"Embedding {len(self.documents)} documents...")
        embeddings = self.model.encode(self.documents, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve_context(self, features_list, k=config.RETRIEVER_TOP_K):
        paper_to_chunks = {}

        """
        Retrieves relevant text snippets for a list of feature names.
        """
        if self.index is None:
            return "Retriever index has not been built."
            
        print("\nQuerying FAISS index for relevant literature context...")
        
        # The features from the explainer are formatted with markdown, clean them first
        clean_features = [f.split(':')[0].replace('*', '').strip() for f in features_list]
        
        query_text = ", ".join(clean_features)
        
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)

        for idx in indices[0]:
            paper_id = self.doc_ids[idx+1]
            chunk_text = self.documents[idx+1]
            # print(chunk_text)

            if paper_id not in paper_to_chunks:
                paper_to_chunks[paper_id] = []
            paper_to_chunks[paper_id].append(chunk_text)
            # assign numeric citation IDs
        
        unique_papers = list(paper_to_chunks.keys())
        citation_map = {i + 1: unique_papers[i] for i in range(len(unique_papers))}

        # build the retrieved context with labeled evidence
        evidence_blocks = []
        for num, paper_id in citation_map.items():
            for ch in paper_to_chunks[paper_id]:
                cleaned = re.sub(r"\s+", " ", ch).strip()
                if cleaned:
                    evidence_blocks.append(f"[{num}] {cleaned}")

        retrieved_context = "\n".join(evidence_blocks)
        print("[Retrieval] Evidence and citation map prepared.")

        return retrieved_context, citation_map

        # retrieved_docs = [self.documents[i] for i in indices[0]]
        # retrieved_ids = [self.doc_ids[i] for i in indices[0]]
        
        # print(f"Retrieved top {k} documents based on features: {retrieved_ids}")
        
        # # For simplicity, we concatenate the full text of the top k documents.
        # # A more advanced approach would be to chunk and retrieve specific sentences.
        # all_retrieved_text = "\n\n---\n\n".join(retrieved_docs)
        
        # return all_retrieved_text

# --- Helper function to simulate text extraction ---
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

def retrieve_top_papers(papers, query_text, model, index, k):
    """
    Retrieves top relevant papers for each input feature and
    builds both:
      - retrieved_context: text snippets with numeric [n] labels
      - citation_map: {n: paper_id}, used later to verify citations

    Requires:
      documents[i]: text chunk i
      doc_ids[i]: source identifier (filename / DOI / etc.) for chunk i
    """
    documents = []
    doc_ids = []
    for name, content in papers.items():
      documents.append(content)
      doc_ids.append(name)

    if index is None:
        return "Retriever not built.", {}

    print("\n[Retrieval] Collecting evidence for citation...")

    # collect chunks by paper_id
    paper_to_chunks = {}

    for raw_feature in query_text.split(","):
        feature = raw_feature.strip()
        if not feature:
            continue

        query_embedding = model.encode([feature])
        distances, indices = index.search(
            np.array(query_embedding).astype('float32'), k
        )

        for idx in indices[0]:
            paper_id = doc_ids[idx]
            chunk_text = documents[idx]

            if paper_id not in paper_to_chunks:
                paper_to_chunks[paper_id] = []
            paper_to_chunks[paper_id].append(chunk_text)

    # assign numeric citation IDs
    unique_papers = list(paper_to_chunks.keys())
    citation_map = {i + 1: unique_papers[i] for i in range(len(unique_papers))}

    # build the retrieved context with labeled evidence
    evidence_blocks = []
    for num, paper_id in citation_map.items():
        for ch in paper_to_chunks[paper_id]:
            cleaned = re.sub(r"\s+", " ", ch).strip()
            if cleaned:
                evidence_blocks.append(f"[{num}] {cleaned}")

    retrieved_context = "\n".join(evidence_blocks)
    print("[Retrieval] Evidence and citation map prepared.")

    return retrieved_context, citation_map