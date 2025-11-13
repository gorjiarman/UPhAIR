# UPhAIR Project

Clinical adoption of machine learning (ML) in medical imaging is limited by the lack of interpretability. To address this, we present UPhAIR (Understandable Post-hoc AI Reports), a pipeline designed to generate transparent, evidence-based explanations by combining SHAP analysis with retrieval-augmented generation (RAG) and large language models (LLMs). We trained 12 Classifier to predict IDH mutation status in glioma using radiomics and clinical features. SHAP values were used to identify key contributors to each prediction. Relevant literature was retrieved from a curated PubMed dataset using FAISS for similarity search and passed through a Retrieval-Augmented Generation (RAG) framework with Gemini 2.5 Pro to generate concise, reference-supported explanations for each feature. The model achieved a best AUROC of 86.7% on a hold-out test set using Gradient Boosting Classifier. In a case study of a single patient excluded from training, the model predicted IDH-wildtype glioma. SHAP identified MGMT status, age, and three radiomic features as most influential. UPhAIR produced a structured report combining SHAP visualizations with LLM-generated summaries grounded in scientific evidence. UPHAIR provides a practical, model-agnostic framework that enhances ML interpretability in clinical settings, helping bridge the gap between black-box AI and real-world medical decision-making. This repository allows you to manage and customize prediction problems easily while automatically handling relevant research papers.

---

## Features

- Fully tested on **Python 3.12.4**.
- Easily customizable prediction problems:
  - Change features and outcomes in the `features` file located in the `data` folder.
- Paper management:
  - Automatically download papers based on your query in the `config` file.
  - Alternatively, you can manually download papers relevant to your prediction problem.
- Inline reporting: Generate clinical interpretation reports directly as PDF.

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd uphair
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have Python **3.12.4** installed.

---

## Configuration

- **Data folder:**  
  The `data/features` file controls which features and outcomes are used for prediction. Edit this file to adjust the model inputs and targets.

- **Config file:**  
  Set your query for paper downloads in the `config` file. The system can automatically fetch relevant research papers based on your query.

---

## Running the Project

1. Prepare the data and configure features.
2. Set your query in `config` (if you want automatic paper downloads).
3. Run the prediction script:

```bash
python main.py
```

*(Replace `main.py` with the entry script of your project.)*

---

## Report

The clinical interpretation report is included here:

<embed src="./reports/clinical_interpretation_report.pdf" type="application/pdf" width="100%" height="800px" />

---

## Notes

- Make sure your environment matches Python 3.12.4 to avoid compatibility issues.
- The project is modular, allowing you to change features, outcomes, and paper queries without modifying the core code.
- All dependencies are listed in `requirements.txt`.

---

## License

[Specify your license here, e.g., MIT License]
