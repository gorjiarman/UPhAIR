import os
import base64
import markdown
from datetime import datetime
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import google.generativeai as genai
import config
from xhtml2pdf import pisa
import base64
import io
from datetime import datetime
from xhtml2pdf import pisa
from PIL import Image
import numpy as np
import requests
class ReportGenerator:
    """
    Generates the final PDF report by combining model output, SHAP explanations,
    and LLM-generated text.
    """
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"

        self.headers = {
            "Authorization": f"apikey {self.api_key}",
            "Content-Type": "application/json"
        }

        # Simple connectivity test
        try:
            payload = {
                "model": config.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "temperature": 0.0
            }

            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            print("🔗 LLM endpoint configured successfully.")
            self.available = True

        except Exception as e:
            print(f"Error configuring LLM endpoint: {e}")
            self.available = False
   
    def build_llm_prompt_assmbly(self, retrieved_context, predicted_outcome_0, features_formatted_0):
        """
        Builds the complete instruction prompt for the clinical report generator.
        Includes examples for:
        - IDH-wildtype (radiomic dominant)
        - IDH-mutant (radiomic dominant)
        - Clinical-feature dominant
        - Mixed (clinical + radiomic)
        """
        return f"""
        You are a medical AI assistant trained to explain how a machine learning model made a specific prediction for a patient based on the feature.
        Your explanation must be clear and readable for physicians and clinicians, using clinical reasoning supported by relevant literature provided below.
        Do not include any treatment recommendations or management advice (e.g., do not say we should use drug X because of gene Y).
        Only describe the clinical or biological relevance of the feature in relation to the predicted class.
        strictly follow the structure and tone of the examples provided.

        Your instructions:
        - Dont bring any additional introduction and explanation before **feature**
        - Explain the model’s prediction based on the given feature realtive value and their SHAP values.
        - For each key feature:
            * when the outcome is IDH-mutant do not explain features with positive shap value
            * when the outcome is IDH-wildtype do not explain features with negetive shap value
            * Report its value and SHAP contribution 
                (SHAP : positive value) --> moved the prediction toward IDH-wildtype
                (SHAP : negetive value) --> moved the prediction toward IDH-mutant
            * Max three sentences
            * Provide a clinical or biological interpretation of its role in the model’s decision.
            * Do not include features that are not explainable.
            * [Relative values] indicate how the feature compares to the training cohort:
                - (Relative Value : 1) : Very Low (below 25th percentile)
                - (Relative Value : 2) : Low (25th to 50th percentile)
                - (Relative Value : 3) : High (50th to 75th percentile)
                - (Relative Value : 4) : Very High (above 75th percentile)
            * Binary features should be reported as 
                "1.00" --> "Present" 
                "0.00" --> "Absent"
            * Support relationships between features and {predicted_outcome_0} using evidence from below Literature.
        - All features except clinical features are derived from radiomic analysis of MRI scans.
        - if an explanation is in contrast with the litrature then return (*) as output

        =========================
        EXAMPLE 1 where the predicted outcome is IDH-Wildtype, age value is 85, and SHAP value is positive and Relative Value is 4:
        **Patient age** which is very high (relative value : 4) compared to the training dataset moved the prediction toward IDH-Wildtype; Higher age at diagnosis is consistently observed in gliomas classified as IDH‑wildtype compared with those bearing IDH‑mutation—for example, one large cohort reported median ages of ~60.5 years for IDH-wildtype versus ~38.2 years for IDH-mutant gliomas (p < 0.001).\n
        EXAMPLE 2 where the predicted outcome is IDH-Mutant, MGMT value is 0.00, and SHAP value is positive and Relative Value is binary:
        **MGMT methylation** status which is absent moved the prediction toward IDH-Mutant; MGMT promoter methylation often appears as an important predictor of IDH mutation in machine learning models because both reflect the G-CIMP epigenetic phenotype typical of lower-grade gliomas.\n
        EXAMPLE 3 where the predicted outcome is IDH-Wildtype, First order minimum value is 10, and SHAP value is positive and Relative Value is 1:
        **First order minimum** which is very low (relative value : 1) compared to the training dataset moved the prediction toward IDH-Wildtype; The first-order minimum in radiomics is the lowest voxel intensity within a tumor ROI, often reflecting necrotic or non-enhancing regions. Lower values are more common in aggressive IDH-wildtype gliomas compared to IDH-mutant tumors.\n
        EXAMPLE 4 where the predicted outcome is IDH-Mutant, Size Zone Non-Uniformity Normalized is 10 and SHAP value is positive and Relative Value is 1:
        **Size Zone Non-Uniformity Normalized** which is very low (relative value : 1) compared to the training dataset moved the prediction toward IDH-Mutant; it captures the variability in the sizes of homogeneous intensity zones within a tumor, with lower values reflecting less internal heterogeneity, patterns that are characteristic of IDH-mutant tumors.\n
        =========================

        **Retrieved Evidence from Literature:**
        ---
        {retrieved_context}
        ---
        
        Feature set:
        {features_formatted_0}
        """        

    def generate_llm_explanation(self, prompt):
        
        if not self.available:
            raise RuntimeError("LLM endpoint is not available.")

        payload = {
            "model": config.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0

        }

        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def create_pdf_report(self, sample_info, prediction_info, references, shap_plot_path, llm_text,img_first,img_mid,img_last):
        """
        Compiles all information into an HTML template and converts it to a PDF.
        """
        print("Compiling the final PDF report...")
        report_path = os.path.join(config.REPORTS_DIR, config.REPORT_FILENAME)

        def encode_np_image(np_img):
            pil = Image.fromarray(np_img)
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        # Encode the 3 slices
        first_b64 = encode_np_image(img_first)
        mid_b64   = encode_np_image(img_mid)
        last_b64  = encode_np_image(img_last)
        # SHAP image encode
        try:
            with open(shap_plot_path, "rb") as f:
                shap_base64 = base64.b64encode(f.read()).decode()
            shap_html = f'<img src="data:image/png;base64,{shap_base64}" style="width: 120px;height: 120px;object-fit: cover;" >'
        except FileNotFoundError:
            shap_html = "<p><strong>Error: SHAP plot image not found.</strong></p>"
        
        print(references)
        
        # HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <h1 style="text-align: center; font-size: 20px;">Machine Learning Prediction Report</h1>
            <style>
                .llm-text {{font-family: "Inter", "Segoe UI", Arial, sans-serif;font-size: 14px;line-height: 1.6;}}
                body {{ font-family: sans-serif; padding-bottom: 0px;}}
                h1 {{ color: #333;  }}
                h2 {{ color: #ffffff; border-bottom: 4px solid #d9534f; padding-bottom: 5px !important; padding-top: 15px; padding-left: 20px; background-color: #333; font-size: 14px; margin-bottom :0px;}}
                table {{ width: 200px; margin-bottom: 5px; }}
                td {{ padding: 4px; border: 2px solid #ddd; }}
                .header {{ background-color: #f2f2f2; font-weight: bold; }}
                .prediction {{ font-size: 24px; color: #d9534f; font-weight: bold; margin-top: 20px; margin-bottom: 0px;}}
                .report-content {{ margin-top: 30px; }}
                .gallery {{display: flex;justify-content: center;align-items: center;gap: 50px;}}
                .gallery img {{width: 120px;height: 120px;object-fit: cover; /* keeps the crop nice */border-radius: 6px; /* optional */}}
            </style>
        </head>
        <body>
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
                Final Prediction: <i>{prediction_info.get('predicted_class', 'N/A')}</i>
            </div>
            <div class="report-content">
                <h2>Clinical Interpretation</h2>
                <div class="llm-text">
                    {markdown.markdown(llm_text)}
                </div>
                <h2>Prediction Explanation (SHAP Analysis) & Tumor Segmentation Slices</h2>
                <div class="gallery">
                    <img src="data:image/png;base64,{first_b64}">
                    <img src="data:image/png;base64,{mid_b64}">
                    <img src="data:image/png;base64,{last_b64}">
                    <img src="data:image/png;base64,{shap_base64}" style="width: 240px;height: 120px;object-fit: cover;" >
                </div>
                <p>Features pushing the prediction higher (towards IDH-wildtype) are red; lower (towards IDH-mutant) are blue.</p>
                <h2>References</h2>
                {markdown.markdown(references)}
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
                image_path = os.path.join(config.REPORTS_DIR, "report_preview_wt.png")
                images[0].save(image_path, "PNG")
                print(f"   Report preview image saved to: {image_path}")
        except Exception as e:
            print(f"Could not create PDF preview image. Poppler might be missing. Error: {e}")