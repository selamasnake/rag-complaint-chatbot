import gradio as gr
from src.rag_pipeline import RAGPipeline

# Initialize RAG pipeline once at startup
rag = RAGPipeline(
    index_path="vector_store/index.faiss",
    metadata_path="vector_store/metadata.pkl"
)


def ask_question(question: str):
    """
    Handle user query, run RAG pipeline, and format output.
    """
    if not question.strip():
        return "Please enter a question.", ""

    answer, sources = rag.run(question)


    source_text = ""
    for i, src in enumerate(sources, 1):
        source_text += (
            f"Source {i}\n"
            f"Product: {src.get('product')}\n"
            f"Complaint ID: {src.get('complaint_id')}\n"
            f"Text: {src.get('text')[:500]}...\n\n"
        )

    return answer, source_text


with gr.Blocks(title="CrediTrust Complaint Analyzer") as demo:
    gr.Markdown(
        """
        # 🏦 CrediTrust Complaint Analyzer

        Ask questions about customer complaints and receive
        evidence-based answers grounded in real complaint data.
        """
    )

    question_input = gr.Textbox(
        label="Enter your question",
        placeholder="Why are customers unhappy with credit cards?",
        lines=2
    )

    ask_button = gr.Button("Ask")

    answer_output = gr.Textbox(
        label="AI-Generated Answer",
        lines=6
    )

    sources_output = gr.Textbox(
        label="Sources Used (Retrieved Complaint Excerpts)",
        lines=10
    )

    clear_button = gr.Button("Clear")

    ask_button.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    clear_button.click(
        fn=lambda: ("", "", ""),
        inputs=None,
        outputs=[question_input, answer_output, sources_output]
    )

demo.launch(share=True)
