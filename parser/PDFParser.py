from llama_cloud_services import LlamaParse

class PDFParser:
    def __init__(self, api_key: str):
        self.parser = LlamaParse(
            api_key=api_key,
            num_workers=4,
            verbose=True,
            language="en",
        )

    def parse_pdf(self, pdf_url: str) -> list:
        result = self.parser.parse(pdf_url)
        markdown_docs = result.get_markdown_documents(split_by_page=True)
        # markdown_docs = result.get_markdown_documents(split_by_page=False)
        return markdown_docs
