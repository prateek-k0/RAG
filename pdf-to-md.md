Got it ✅ — you want to parse a PDF into Markdown using the marker-pdf Python package.
marker-pdf is a high-accuracy document converter that supports PDF → Markdown, JSON, HTML, and more 1.
Here’s a complete, runnable Python example that:

Installs and imports marker-pdf
Converts a PDF to Markdown
Handles errors and validates inputs


Python Code
```python
Pythonimport os
import subprocess
import sys
from pathlib import Path

def convert_pdf_to_markdown(pdf_path, output_dir="output_md"):
    """
    Convert a PDF file to Markdown using marker-pdf.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save the Markdown output.
    """
    # Validate PDF path
    pdf_file = Path(pdf_path)
    if not pdf_file.exists() or pdf_file.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Invalid PDF file: {pdf_path}")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Run marker-pdf CLI for single file conversion
        subprocess.run(
            [
                sys.executable, "-m", "marker.convert_single",
                str(pdf_file),
                "--output_format", "markdown",
                "--output_dir", str(output_path)
            ],
            check=True
        )
        print(f"✅ Conversion complete! Markdown saved in: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during conversion: {e}")
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")

if __name__ == "__main__":
    # Example usage
    # Make sure marker-pdf is installed: pip install marker-pdf
    pdf_input = "sample.pdf"  # Change to your PDF path
    convert_pdf_to_markdown(pdf_input)
```

Steps to Run


Install marker-pdf (Python 3.10+ required)
```sh
pip install marker-pdf
```

If you want support for DOCX, PPTX, images, etc.:
```sh
pip install "marker-pdf[full]"
```



Save the script above as `pdf_to_md.py`


Run the script:
```sh
python pdf_to_md.py
```


The converted Markdown file will be in the `output_md/` folder.



Extra Options (from marker-pdf CLI) 1
You can pass additional flags to improve accuracy:

`--use_llm` → Use an LLM to fix tables, math, and formatting.
`--page_range` "0,5-10" → Convert only specific pages.
`--force_ocr` → Force OCR for scanned PDFs.
`--disable_image_extraction` → Skip image extraction.

Example with LLM and page range:
Bash
```sh
python -m marker.convert_single mydoc.pdf --output_format markdown --use_llm --page_range "1-5"
```
