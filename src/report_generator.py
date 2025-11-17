import os
import base64
import markdown
from datetime import datetime
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import google.generativeai as genai
import config
from xhtml2pdf import pisa

class ReportGenerator:
    """
    Generates the final PDF report by combining model output, SHAP explanations,
    and LLM-generated text.
    """
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.llm_model = genai.GenerativeModel(config.LLM_MODEL_NAME)
            print("Google Generative AI configured successfully.")
        except Exception as e:
            print(f"Error configuring Google Generative AI: {e}")
            self.llm_model = None
    
    def validate_report_citations(self, report_text, citation_map):
        """
        Verifies that every [n] in the report matches a retrieved source in citation_map.

        Returns:
            all_citations_valid: bool
            used_citations: list[int] actually cited
            invalid_citations: list[int] not present in citation_map
            unreferenced_sources: list[int] retrieved but not cited
        """
        import re

        # find all [n] citations
        cited_nums = re.findall(r"\[(\d+)\]", report_text)
        cited_nums = sorted({int(x) for x in cited_nums})

        valid_nums = sorted(citation_map.keys())

        invalid = [n for n in cited_nums if n not in citation_map]
        unref = [n for n in valid_nums if n not in cited_nums]

        return {
            "all_citations_valid": (len(invalid) == 0),
            "used_citations": cited_nums,
            "invalid_citations": invalid,
            "unreferenced_sources": unref,
            "citation_lookup": {n: citation_map[n] for n in cited_nums if n in citation_map}
        }

    def build_llm_prompt(self, retrieved_context, predicted_outcome_0, features_formatted_0):
        """
        Builds the complete instruction prompt for the clinical report generator.
        Includes examples for:
        - IDH-wildtype (radiomic dominant)
        - IDH-mutant (radiomic dominant)
        - Clinical-feature dominant
        - Mixed (clinical + radiomic)
        """
        return f"""
        You are a medical AI assistant trained to generate precise and formal clinical interpretation reports.
        Your goal is to explain how a machine learning model made a specific prediction for a patient based on their input features.
        Your explanation must be clear and readable for physicians and clinicians, using clinical reasoning supported by relevant literature provided below.
        Do not include any treatment recommendations or management advice (e.g., do not say we should use drug X because of gene Y).
        Only describe the clinical or biological relevance of the feature in relation to the predicted class.

        **Retrieved Evidence from Literature:**
        ---
        {retrieved_context}
        ---

        Your instructions:
        - Explain the model’s prediction based on the given features and their SHAP values.
        - Prioritize features with higher absolute SHAP values (positive or negative).
        - For each key feature:
            * Use a new paragraph
            * Report its value and SHAP contribution.
            * Provide a clinical or biological interpretation of its role in the model’s decision.
            * Do not include features that are not explainable.
            * Support relationships between features and {predicted_outcome_0} using ONLY the research papers provided above as in-context evidence.
        - Cite each reference inline using numbers in square brackets (e.g., [1], [2]).
        - Keep the explanation for each feature under 3 sentences.
        - At the end, include a reference list with:
        - Do not reference any papers not included in the retrieved context.
            * Title of each cited paper.
            * First author's name.

        Always include these four lines exactly:
        Explanation:

        The model predicted this patient to be {predicted_outcome_0}
        
        Several features influenced this prediction:

        =========================
        EXAMPLE 1:
        **Age (65 years) contributed positively;** IDH-wildtype gliomas are more frequent in older patients [1].\n

        **Tumor volume was large (58 cc);** supports a more aggressive tumor subtype [2].\n

        **GLCM entropy was high;** indicating structural heterogeneity, common in IDH-wildtype cases [3].\n

        **MGMT methylation decreased the score;** as methylation is more associated with IDH-mutant tumors [4].\n
        
        **Tumor location in the frontal lobe also decreased the prediction;** consistent with the IDH-mutant phenotype [5].\n

        References:
        [1] The Relationship Between Age and Glioma Subtypes – Smith
        [2] Imaging Biomarkers of Glioma Aggressiveness – Tanaka
        [3] Texture Analysis for Glioma Characterization – Alvarez
        [4] MGMT Methylation and Glioma Molecular Subtypes – Zhang
        [5] Spatial Distribution of Glioma Types in the Brain – Cho
        =========================

        EXAMPLE 2:
        **GLSZM ZoneEntropy was low;** lower textural heterogeneity is described in IDH-mutant gliomas [1].\n

        **High sphericity of the enhancing lesion;** more regular tumor shape is linked to less infiltrative phenotypes [2].\n

        **Lower peritumoral edema volume;** frequently observed in IDH-mutant tumors [3].\n

        References:
        [1] Texture-Based Stratification of Glioma Subtypes – Liu
        [2] Morphological Signatures in Lower-Grade Glioma – Patel
        [3] Edema Patterns and Molecular Class in Glioma – Rossi
        =========================

        Now generate the actual report for THIS patient, following the same structure and tone.
        Use only the provided evidence, and do not invent new biological claims.


        Feature set:
        {features_formatted_0}
        """
    
    def generate_llm_explanation(self, prompt):
        """
        Prompts the LLM to generate a clinical interpretation.
        """
        if not self.llm_model:
            return "LLM was not configured due to an API key error."

        print("\nGenerating clinical narrative with the LLM...")
        
        try:
            response = self.llm_model.generate_content(prompt)
            print("LLM narrative generated successfully.")
            return response.text
        except Exception as e:
            print(f"Error during LLM content generation: {e}")
            return f"Error generating report: {e}"

    def create_pdf_report(self, sample_info, prediction_info, shap_plot_path, llm_text):
        """
        Compiles all information into an HTML template and converts it to a PDF.
        """
        print("Compiling the final PDF report...")
        report_path = os.path.join(config.REPORTS_DIR, config.REPORT_FILENAME)

        # Encode the SHAP plot image to embed it in the HTML
        try:
            with open(shap_plot_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            image_html = f'<img src="data:image/png;base64,{img_base64}" style="width: 50%; max-width: 300px; margin-top: 20px;" >'
        except FileNotFoundError:
            image_html = "<p><strong>Error: SHAP plot image not found.</strong></p>"
        
        # Convert the LLM's markdown response to HTML
        report_html = markdown.markdown(llm_text)
        
        # HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Machine Learning Prediction Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 2px solid #f0f0f0; padding-bottom: 5px;}}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                td {{ padding: 8px; border: 1px solid #ddd; }}
                .header {{ background-color: #f2f2f2; font-weight: bold; }}
                .prediction {{ font-size: 24px; color: #d9534f; font-weight: bold; margin-top: 20px; }}
                .report-content {{ margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>Machine Learning Prediction Report</h1>
            <h2>Glioma IDH Classification</h2>
            
            <table>
                <tr>
                    <td class="header">Patient ID:</td>
                    <td>{sample_info.get('PatientID', 'N/A')}</td>
                    <td class="header">Patient Age:</td>
                    <td>{sample_info.get('Age', 'N/A')}</td>
                </tr>
                <tr>
                    <td class="header">Report Date:</td>
                    <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                    <td class="header">Model Used:</td>
                    <td>{prediction_info.get('model_name', 'N/A')}</td>
                </tr>
            </table>

            <div class="prediction">
                Final Prediction: {prediction_info.get('predicted_class', 'N/A')}
            </div>

            <div class="report-content">
                <h2>Clinical Interpretation</h2>
                {report_html}

                <h2>Prediction Explanation (SHAP Analysis)</h2>
                <p>The following plot shows the features that contributed most to the model's prediction for this patient. Features pushing the prediction higher (towards IDH-wildtype) are in red, and those pushing it lower (towards IDH-mutant) are in blue.</p>
                {image_html}
            </div>
        </body>
        </html>
        """
        
        # Generate PDF
        with open(report_path, "wb") as f:
            pisa.CreatePDF(html_template, dest=f)
        print(f"Report successfully saved to: {report_path}")

        # Optional: Convert PDF to image for display
        try:
            images = convert_from_path(report_path, dpi=150)
            if images:
                image_path = os.path.join(config.REPORTS_DIR, "report_preview.png")
                images[0].save(image_path, "PNG")
                print(f"   Report preview image saved to: {image_path}")
        except Exception as e:
            print(f"Could not create PDF preview image. Poppler might be missing. Error: {e}")