from transformers import pipeline
from .prompts import RAG_PROMPT


class AnswerGenerator:
    """
    Generates answers grounded in retrieved complaint context.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        max_new_tokens: int = 300,
        temperature: float = 0.1
    ):
        self.llm = pipeline(
            "text2text-generation",  
            model=model_name
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, documents: list[str]) -> str:
        if not documents:
            return "No relevant complaint data found."

        # Keep only top 3 chunks
        context = "\n\n".join(documents[:3])

        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        output = self.llm(
            prompt,
            max_new_tokens=self.max_new_tokens
        )[0]["generated_text"]

        return output.strip()

