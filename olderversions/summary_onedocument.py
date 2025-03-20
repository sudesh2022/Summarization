import ollama
import PyPDF2
import concurrent.futures
import time

def query_ollama(model: str, prompt: str):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.get("message", {}).get("content", "No response")
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return ""

def read_pdf(file_path: str) -> list:
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text_chunks = [page.extract_text() for page in reader.pages if page.extract_text()]
    print(f"Extracted {len(text_chunks)} text chunks from PDF.")
    return text_chunks

def summarize_chunk(chunk: str, model: str) -> str:
    print("Summarizing chunk...")
    prompt = f"Summarize the following text concisely:\n{chunk}"
    return query_ollama(model, prompt)

def map_reduce_summary(chunks: list, model: str) -> str:
    summaries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # Limit threads
        future_to_chunk = {executor.submit(summarize_chunk, chunk, model): chunk for chunk in chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                summaries.append(future.result())
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    print("Combining partial summaries...")
    combined_text = "\n".join(summaries)
    time.sleep(2)  # Small delay to prevent overloading Ollama
    final_summary = summarize_chunk(combined_text, model)  # Reduce step
    return final_summary

if __name__ == "__main__":
    model_name = "mistral"  # Change to the Ollama model you want to use
    pdf_path = "document.pdf"  # Change to your PDF file path
    
    text_chunks = read_pdf(pdf_path)
    if not text_chunks:
        print("No text found in PDF.")
    else:
        summary = map_reduce_summary(text_chunks, model_name)
        print("Final Summary:", summary)
