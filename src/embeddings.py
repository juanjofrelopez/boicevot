from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader


class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", max_workers=4):
        self.model = SentenceTransformer(model_name)
        self.max_workers = max_workers

    def _process_txt_line(self, line):
        return line, self.model.encode(line)
    
    def generate_embeddings_from_pdf(self, file_path):
        try:
            all_lines = []
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    all_lines.extend(lines)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._process_txt_line, all_lines))
                return results
        except Exception as e:
            print(f"Error reading or processing the file: {e}")
            return None

    def generate_embeddings_from_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                lines = [line for line in content.splitlines() if line.strip()]
                print("got", len(lines), "lines of content")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(self._process_txt_line, lines))
                    return results
        except Exception as e:
            print(f"Error reading or processing the file: {e}")
            return None

    def generate_embeddings_from_string(self, query: str):
        return self.model.encode(query)
