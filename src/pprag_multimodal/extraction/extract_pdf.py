import os
import json
import logging
import zipfile
import shutil
from pathlib import Path

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
from pprag_multimodal.config import DATASET_DIR, PDF_DIR

logger = logging.getLogger(__name__)
# Adobe Extract API currently places rendered assets under these ZIP prefixes.
ADOBE_ASSET_PREFIXES = ("figures/", "tables/")
_credentials_cache = None

def get_credentials():
    global _credentials_cache
    if _credentials_cache is not None:
        return _credentials_cache
    client_id = os.environ.get("ADOBE_CLIENT_ID")
    client_secret = os.environ.get("ADOBE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Adobe credentials missing from environment or .env")
    _credentials_cache = ServicePrincipalCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return _credentials_cache

def extract_pdf_to_md(pdf_path: str, output_dir: str):
    pdf_name = Path(pdf_path).stem
    paper_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(paper_dir, exist_ok=True)

    zip_path = os.path.join(paper_dir, "extract.zip")
    tmp_zip_path = f"{zip_path}.tmp"
    md_path = os.path.join(paper_dir, f"{pdf_name}.md")

    # If already extracted, skip Adobe API call
    if not os.path.exists(zip_path):
        logger.info(f"Extracting {pdf_name} using Adobe SDK...")
        try:
            credentials = get_credentials()
            pdf_services = PDFServices(credentials=credentials)

            with open(pdf_path, 'rb') as f:
                input_stream = f.read()

            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Configure extraction params for text, tables, and figures
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT],
                elements_to_extract_renditions=[ExtractRenditionsElementType.TABLES, ExtractRenditionsElementType.FIGURES]
            )

            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            with open(tmp_zip_path, "wb") as f:
                f.write(stream_asset.get_input_stream())
            os.replace(tmp_zip_path, zip_path)
        except Exception:
            if os.path.exists(tmp_zip_path):
                os.remove(tmp_zip_path)
            raise

    logger.info(f"Processing ZIP file for {pdf_name}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        base_dir = Path(paper_dir).resolve()
        # Extract images from zip into a subfolder
        for member in zip_ref.namelist():
            if any(member.startswith(prefix) for prefix in ADOBE_ASSET_PREFIXES):
                target_path = (base_dir / member).resolve()
                if target_path != base_dir and base_dir not in target_path.parents:
                    logger.warning("Skipping suspicious ZIP member: %s", member)
                    continue
                zip_ref.extract(member, paper_dir)

        # Read the structural JSON
        if "structuredData.json" not in zip_ref.namelist():
            raise RuntimeError(f"structuredData.json not found in Adobe extraction result: {zip_path}")
        with zip_ref.open("structuredData.json") as json_file:
            data = json.load(json_file)

    markdown_lines = []

    for element in data.get("elements", []):
        path = element.get("Path", "")
        text = element.get("Text", "").strip()

        # Check if there are associated file renditions (images/tables)
        if "filePaths" in element and len(element["filePaths"]) > 0:
            for file_path in element["filePaths"]:
                # The file might be in figures/ or tables/ folder
                # We format it as a markdown image reference
                markdown_lines.append(f"\n![{text}]({file_path})\n")
            continue

        if not text:
            continue

        if "/H1" in path:
            markdown_lines.append(f"\n# {text}\n")
        elif "/Title" in path:
            markdown_lines.append(f"\n# {text}\n")
        elif "/H2" in path:
            markdown_lines.append(f"\n## {text}\n")
        elif "/H3" in path:
            markdown_lines.append(f"\n### {text}\n")
        elif "/H4" in path:
            markdown_lines.append(f"\n#### {text}\n")
        elif "/P" in path:
            markdown_lines.append(f"\n{text}\n")
        elif "/LBody" in path or "/LI" in path:
            markdown_lines.append(f"- {text}")
        else:
            markdown_lines.append(f"{text}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))

    logger.info(f"Markdown and images generated in {paper_dir}")

def process_all_pdfs():
    pdf_dir = Path(PDF_DIR)
    output_dir = Path(DATASET_DIR)

    os.makedirs(output_dir, exist_ok=True)

    failed_files = []
    succeeded = 0
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            logger.info("Processing PDF: %s", pdf_file)
            extract_pdf_to_md(str(pdf_file), str(output_dir))
            succeeded += 1
        except Exception as exc:
            logger.exception("Failed to process %s", pdf_file)
            failed_files.append((str(pdf_file), str(exc)))
    logger.info("PDF extraction complete: %d succeeded, %d failed", succeeded, len(failed_files))
    if failed_files:
        for file_name, error in failed_files:
            logger.error("  %s: %s", file_name, error)

if __name__ == "__main__":
    process_all_pdfs()
