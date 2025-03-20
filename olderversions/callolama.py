import ollama
import PyPDF2

def query_ollama(model: str, prompt: str):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.get("message", {}).get("content", "No response")

def read_pdf(file_path: str) -> str:
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def summarize_text(text: str, model: str):
    prompt = f"Summarize the following text in 500 words:\n{text}"
    return query_ollama(model, prompt)

if __name__ == "__main__":
    model_name = "mistral"  # Change to the Ollama model you want to use
    pdf_path = "ceoguide.pdf"  # Change to your PDF file path
    
    text_content = read_pdf(pdf_path)
    summary = summarize_text(text_content, model_name)
    
    print("Summary:", summary)
