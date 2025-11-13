import config
from data_handler import DataHandler
from model_trainer import ModelTrainer
from explainer import Explainer
from retriever import FaissRetriever, get_paper_texts,PaperRetriever
from report_generator import ReportGenerator

def main():
    """
    Orchestrates the entire UPAIR pipeline.
    """
    
    
    # If you want to download and process papers based on the search queery, uncomment below:
    # print("--- Finding And Downloading Related Context ---")
    # paper_retriver = PaperRetriever()
    # papers_metadata = paper_retriver.fetch_paper_metadata()
    # paper_retriver.download_papers(papers_metadata)
    
    # If you have already downloaded papers in data/papers path, uncomment below:
    # print("--- Extracting Context ---")
    # paper_retriver = PaperRetriever()
    # paper_retriver.extract_text_and_merge()

    print("--- Starting UPAIR Pipeline ---")

    # 1. Load and Prepare Data
    data_handler = DataHandler()
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
    raw_prediction, predicted_class = trainer.predict_sample(data_handler.sample_X)
    prediction_info = {
        'model_name': trainer.best_classifier_name,
        'predicted_class': predicted_class,
        'raw_prediction': raw_prediction
    }

    # 4. Explain the Prediction with SHAP
    explainer = Explainer(trainer.best_model, data_handler.X_train)
    explainer.fit_explainer()
    top_features, shap_plot_path = explainer.explain_sample(data_handler.sample_X)
    
    if top_features is None:
        print("Could not generate SHAP explanations. Exiting.")
        return
    # 5. Build Retriever and Get Context
    # This step simulates downloading and processing papers.
    paper_texts = get_paper_texts()
    paper_texts = {k: v for k, v in paper_texts.items() if v != ""}
    faiss_retriever = FaissRetriever()
    faiss_retriever.build_index_from_texts(paper_texts)
    retrieved_context, citation_map = faiss_retriever.retrieve_context(top_features,5)
    
    for attempt in range(1, config.MAX_ATTEMPT + 1):
        
        report_generator = ReportGenerator(api_key=config.GEMINI_API_KEY)
        print(f"\n=== LLM GENERATION ATTEMPT {attempt}/{config.MAX_ATTEMPT} ===")

        llm_prompt = report_generator.build_llm_prompt(str(retrieved_context),str(predicted_class),str(top_features))

        try:
            candidate_report = report_generator.generate_llm_explanation(llm_prompt)
        except Exception as e:
            print("Gemini generation error:", e)
            candidate_report = f"Error during report generation: {e}"

        # Validate citations
        validation_result = report_generator.validate_report_citations(candidate_report, citation_map)

        if validation_result["all_citations_valid"]:
            print("Citations valid. Report accepted.")
            final_report = candidate_report
            break
        else:
            print("Invalid citations found:", validation_result["invalid_citations"])
        
    
    report_generator.create_pdf_report(
        sample_info=data_handler.sample_info,
        prediction_info=prediction_info,
        shap_plot_path=shap_plot_path,
        llm_text=final_report
    )

    print("\n--- UPAIR Pipeline Finished ---")
if __name__ == '__main__':
    main()
