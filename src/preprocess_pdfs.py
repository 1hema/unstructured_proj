import os
from pypdf import PdfReader
from tqdm import tqdm

PDF_DIR = "raw_data/"
OUTPUT_DIR = "./processed_txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    text_chunks = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text_chunks.append(txt.strip())
    except Exception as e:
        print(f"[WARN] Failed to parse {pdf_path}: {e}")
    return "\n\n".join(text_chunks)

def main():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    for pdf in tqdm(pdf_files, desc="Preprocessing PDFs"):
        pdf_path = os.path.join(PDF_DIR, pdf)
        txt = extract_text_from_pdf(pdf_path)
        if not txt.strip():
            print(f"[WARN] No text extracted from {pdf}")
            continue
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(pdf)[0] + ".txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
