import ollama
from pdfminer.high_level import extract_text
import concurrent.futures
import time
import os
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def query_ollama(model: str, prompt: str):
    """Query Ollama API with a given model and prompt."""
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
    """Extract text from a PDF efficiently using pdfminer."""
    text = extract_text(file_path)
    text_chunks = text.split("\n\n")  # Split into chunks by paragraphs
    print(f"Extracted {len(text_chunks)} text chunks from {file_path}.")
    return text_chunks

def summarize_chunk(chunk: str, model: str) -> str:
    """Summarize a chunk of text using Ollama."""
    print("Summarizing chunk...")
    prompt = f"Summarize the following text concisely:\n{chunk}"
    return query_ollama(model, prompt)

def map_reduce_summary(chunks: list, model: str) -> str:
    """Perform map-reduce summarization using parallel execution."""
    summaries = []

    # Use more threads for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_chunk = {executor.submit(summarize_chunk, chunk, model): chunk for chunk in chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                summaries.append(future.result())
            except Exception as e:
                print(f"Error processing chunk: {e}")

    print("Combining partial summaries...")
    combined_text = " ".join(summaries)  # Use space instead of newline for efficiency
    return summarize_chunk(combined_text, model)  # Reduce step

@app.route("/", methods=["GET", "POST"])
def upload_files():
    summaries = {}
    if request.method == "POST":
        files = request.files.getlist("files")
        for file in files:
            if file.filename.endswith(".pdf"):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                
                # Parallel PDF processing
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    future = executor.submit(read_pdf, file_path)
                    text_chunks = future.result()

                if text_chunks:
                    summaries[file.filename] = map_reduce_summary(text_chunks, "granite3-moe:3b")
                else:
                    summaries[file.filename] = "No text found in PDF."

    return render_template("index.html", summaries=summaries)

if __name__ == "__main__":
    app.run(debug=True)
