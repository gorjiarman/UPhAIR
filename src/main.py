import config
from data_handler import DataHandler
from model_trainer import ModelTrainer
from explainer import Explainer
from retriever import FaissRetriever, get_paper_texts,PaperRetriever
from report_generator import ReportGenerator
from paper_downloader import MultiSourcePaperFetcher
import pandas as pd

def main():
    """
    Orchestrates the entire UPAIR pipeline.
    """
    # If you want to download and process papers based on the search queery, uncomment below:
    # print("--- Finding And Downloading Related Context ---")
    # fetcher = MultiSourcePaperFetcher(unpaywall_email="gorjiarman@gmail.com")
    # fetcher.fetch(query="glioma radiomics", max_results=50)
    
    # If you have already downloaded papers in data/papers path, uncomment below:
    # print("--- Extracting Context ---")
    # paper_retriver = PaperRetriever()
    # paper_retriver.extract_text_and_merge()
    
    print("--- Starting UPAIR Pipeline ---")

    # 1. Load and Prepare Data
    data_handler = DataHandler()
    features_array,feature_names,first_slice_img,middle_slice_img,last_slice_img = data_handler.load_sample()

    if not data_handler.load_data():
        return # Exit if data loading fails
    data_handler.prepare_and_split_data()
    
    # 2. Train Model and Find the Best One
    trainer = ModelTrainer(config.CLASSIFIERS)
    trainer.train_and_evaluate(
        data_handler.X_train, data_handler.y_train,
        data_handler.X_test, data_handler.y_test
    )

    # 3. Make Prediction on a Single Sample
    raw_prediction, predicted_class = trainer.predict_sample(features_array.reshape(1, -1))
    prediction_info = {
        'model_name': trainer.best_classifier_name,
        'predicted_class': predicted_class,
        'raw_prediction': raw_prediction
    }

    # 4. Explain the Prediction with SHAP
    explainer = Explainer(trainer.best_model, data_handler.X_train)
    explainer.fit_explainer()
    top_features, shap_plot_path = explainer.explain_sample(pd.DataFrame.from_records(features_array.reshape(1, -1)))
    
    if top_features is None:
        print("Could not generate SHAP explanations. Exiting.")
        return
    references = []
    # 5. Build Retriever and Get Context
    # This step simulates downloading and processing papers.
    features_int_string = ""
    paper_texts = get_paper_texts()
    paper_texts = {k: v for k, v in paper_texts.items() if v != ""}

    counter = 0
    for top_feature in top_features:
        if counter<5:
            print(top_feature)
            print(predicted_class)
            shap_value = float(
                top_feature.split("SHAP :")[1].split(")")[0]
            )

            if (
                (predicted_class == "IDH-wildtype" and shap_value > 0) or
                (predicted_class == "IDH-mutant" and shap_value < 0)
            ):
                top_feature = top_feature.split(":")[0]
                print(f"\n_______________________________________________\n🫆 Top feature: {top_feature}")
                faiss_retriever = FaissRetriever()
                try:
                    faiss_retriever.build_index_from_texts(feature_name=top_feature, paper_texts=paper_texts)
                    best_paper, full_text = faiss_retriever.retrieve_context(features=top_feature, paper_texts=paper_texts)

                except Exception as e:
                    print("Retriever error:", e)
                report_generator = ReportGenerator(api_key=config.LLM_API_KEY, endpoint=config.END_POINT)
                llm_prompt = report_generator.build_llm_prompt_assmbly(str(full_text),str(predicted_class),str(top_feature))

                try:
                    candidate_report = report_generator.generate_llm_explanation(llm_prompt)
                except Exception as e:
                    print("LLM generation error:", e)
                    candidate_report = f"Error during report generation: {e}"
                
                print(candidate_report)
                if "(*)" not in candidate_report and "**" in candidate_report:
                    features_int_string = features_int_string+"\n"+candidate_report.lstrip()
                    counter=counter+1
                    if best_paper not in references:
                        references.append("\n"+best_paper)
            
    print("=====================================================")
    print(features_int_string)
    refstr = ""
    for refrence in references:
        refstr=refstr+refrence+"\n"
    
    report_generator.create_pdf_report(
        sample_info=data_handler.sample_info,
        prediction_info=prediction_info,
        references=refstr,
        shap_plot_path=shap_plot_path,
        llm_text=features_int_string,
        img_first = first_slice_img,
        img_mid = middle_slice_img,
        img_last= last_slice_img
    )
    
    print("\n--- UPAIR Pipeline Finished ---")
if __name__ == '__main__':
    main()
