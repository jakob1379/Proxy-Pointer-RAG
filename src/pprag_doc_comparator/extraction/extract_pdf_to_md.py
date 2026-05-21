"""
DocComparator: PDF to Markdown Extraction via LlamaParse

Converts PDF documents to structured Markdown files.
Skips PDFs that already have a corresponding .md file.
Includes post-processing to promote legal heading patterns (ARTICLE, SECTION)
into ATX headings for skeleton tree construction.

Usage:
    from pprag_doc_comparator.extraction.extract_pdf_to_md import extract_to_md
"""
import os
import re
import sys
import shutil
import logging

PARSE_VERSION = "v2"

from pprag_doc_comparator.config import LLAMA_PARSE_TIER


def extract_pdf(pdf_path, output_dir):
    """Convert a single PDF to Markdown using LlamaParse."""
    try:
        from llama_cloud import LlamaCloud  # type: ignore
    except ImportError:
        logging.error("llama-cloud not installed. Run: pip install llama-cloud")
        return None

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        logging.error("LLAMA_CLOUD_API_KEY not set in .env")
        return None

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.md")

    if os.path.exists(output_path):
        logging.info(f"  [SKIP] {base_name}.md already exists.")
        return output_path

    logging.info(f"  Extracting: {os.path.basename(pdf_path)}...")

    try:
        client = LlamaCloud(api_key=api_key)

        # Upload, parse, and wait for result in one call
        with open(pdf_path, "rb") as f:
            result = client.parsing.parse(
                upload_file=f,
                tier=LLAMA_PARSE_TIER,
                version=PARSE_VERSION,
                expand=["markdown"],
            )

        # Check for response safety
        if not result.markdown or not result.markdown.pages:
            logging.error("  -> No markdown content found in result.")
            return None

        pages = result.markdown.pages
        full_md = "\n\n".join(
            page.markdown for page in pages
            if getattr(page, "markdown", None)
        )

        # Write markdown output
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(full_md)

        logging.info(f"  -> Saved: {output_path} ({len(pages)} pages, {len(full_md)} chars)")
        return output_path

    except Exception as e:
        logging.error(f"  -> Failed: {e}")
        logging.info("  Check https://docs.cloud.llamaindex.ai/ for API updates.")
        return None


# ── Post-Processing: Heading Promotion ──────────────────────────────────

# SEC page-break noise patterns to skip during TOC parsing
_NOISE_PATTERNS = [
    re.compile(r'^https?://', re.IGNORECASE),
    re.compile(r'^sec\.gov/', re.IGNORECASE),
    re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4}'),       # dates: 5/7/26
    re.compile(r'^[ivxlc]+\s*$', re.IGNORECASE),     # roman page numbers
    re.compile(r'^\d{1,3}\s*$'),                      # bare page numbers
    re.compile(r'^\d{1,3}/\d{1,3}\s*$'),              # page fractions: 2/190
]


def _is_noise(text):
    """Check if a line is SEC page-break noise."""
    return any(p.match(text) for p in _NOISE_PATTERNS)


def _strip_markup(text):
    """Remove HTML tags and markdown bold/italic for comparison."""
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    text = re.sub(r'[*_]+', '', text)
    return text


def _normalize(text):
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r'\s+', ' ', text).strip()


def _extract_body_after_title(original_line, toc_title):
    """
    Given a line like 'SECTION 2.04. <u>Title</u>. Body text...'
    and a TOC title like 'SECTION 2.04. Title',
    find and return the body text portion after the title ends.
    """
    toc_idx = 0
    orig_idx = 0
    toc_len = len(toc_title)
    orig_len = len(original_line)

    while toc_idx < toc_len and orig_idx < orig_len:
        ch = original_line[orig_idx]
        if ch == '<':
            end = original_line.find('>', orig_idx)
            if end != -1:
                orig_idx = end + 1
                continue
        if ch in ('*', '_'):
            orig_idx += 1
            continue
        if ch == toc_title[toc_idx]:
            toc_idx += 1
            orig_idx += 1
        else:
            orig_idx += 1

    if toc_idx >= toc_len:
        remainder = original_line[orig_idx:].strip()
        remainder = remainder.lstrip('.').strip()
        return remainder if remainder else None
    return None


def _classify_toc_entry(text):
    """
    Classify a cleaned TOC text line as an article or section entry.
    Returns (title, 'article'|'section') or None.
    """
    text = text.strip()
    if not text or len(text) < 4:
        return None

    # ARTICLE entry: "ARTICLE 1 DEFINITIONS..." or "Article 5. COVENANTS"
    m = re.match(
        r'^((?:ARTICLE)\s+(?:[IVXLCDM]+|\d+)\.?\s+.+?)(?:\s+\d{1,3})?\s*$',
        text, re.IGNORECASE
    )
    if m:
        return (m.group(1).strip(), 'article')

    # SECTION-prefixed: "Section 5.01 Information 50"
    m = re.match(
        r'^((?:SECTION)\s+\d+[\./\d]*\.?\s+.+?)(?:\s+\d{1,3})?\s*$',
        text, re.IGNORECASE
    )
    if m:
        return (m.group(1).strip(), 'section')

    # Bare numbered: "8.01 Events of Default 93"
    m = re.match(
        r'^(\d+\.\d+[\./\d]*\.?\s+.+?)(?:\s+\d{1,3})?\s*$',
        text
    )
    if m:
        title = m.group(1).strip()
        if len(title) > 5:
            return (title, 'section')

    return None


def _parse_toc(lines, toc_start):
    """
    Parse the full Table of Contents, handling both HTML table format
    and plain text format across multiple pages with SEC noise.

    Returns: (toc_entries, toc_end)
        toc_entries: list of (title, 'article'|'section')
        toc_end: line index where the TOC ends
    """
    toc_entries = []
    toc_end = None

    for i in range(toc_start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue

        # Stop at Schedules/Exhibits (end of TOC)
        if re.match(r'^(SCHEDULES|EXHIBITS)\s*:?\s*$', stripped, re.IGNORECASE):
            toc_end = i
            break

        # Skip noise
        if _is_noise(stripped):
            continue

        # Skip HTML structural tags (but process <td> content)
        td_match = re.search(r'<td[^>]*>(.*?)</td>', stripped, re.IGNORECASE)
        if td_match:
            cell_text = re.sub(r'\*\*', '', td_match.group(1)).strip()
            result = _classify_toc_entry(cell_text)
            if result:
                toc_entries.append(result)
            continue

        # Skip other HTML tags
        if stripped.startswith('<'):
            continue

        # Try plain text line
        result = _classify_toc_entry(stripped)
        if result:
            toc_entries.append(result)

    if toc_end is None:
        toc_end = min(toc_start + 500, len(lines) - 1)

    return toc_entries, toc_end


def _promote_headings(md_path):
    """
    Post-process extracted markdown to promote legal heading patterns
    into ATX headings using the document's Table of Contents as reference.

    Handles:
    - HTML table TOC entries (<td>...</td>)
    - Plain text TOC entries (multi-page TOCs with SEC page-break noise)
    - ARTICLE and SECTION prefixed entries
    - Bare numbered entries (e.g. "8.01 Events of Default")
    - ARTICLE headings split across two lines in the body
    - False LlamaParse headings like "# (a)" demoted to plain text

    Safe to re-run: strips prior ATX prefixes before re-promoting.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    # ── Phase 1: Find the TOC ───────────────────────────────────────
    toc_start = None
    for i, line in enumerate(lines):
        clean = re.sub(r'^#{1,6}\s+', '', line.strip())
        if re.match(r'^TABLE\s+OF\s+CONTENTS\s*$', clean, re.IGNORECASE):
            toc_start = i
            break

    if toc_start is None:
        return _promote_headings_basic(md_path, lines)

    # ── Phase 2: Parse full TOC ─────────────────────────────────────
    toc_entries, toc_end = _parse_toc(lines, toc_start)

    if not toc_entries:
        return _promote_headings_basic(md_path, lines)

    # Sort longest first to prevent prefix collisions
    toc_entries.sort(key=lambda x: len(x[0]), reverse=True)

    # Build ARTICLE-only lookup for split-heading matching
    # Maps "ARTICLE 1" -> "ARTICLE 1 DEFINITIONS AND ACCOUNTING TERMS"
    article_lookup = {}
    for title, level in toc_entries:
        if level == 'article':
            m = re.match(r'^(ARTICLE\s+(?:[IVXLCDM]+|\d+))\.?', title, re.IGNORECASE)
            if m:
                article_lookup[_normalize(m.group(1)).upper()] = title

    n_articles = sum(1 for _, l in toc_entries if l == 'article')
    n_sections = sum(1 for _, l in toc_entries if l == 'section')
    logging.info(f"  [TOC] Found {len(toc_entries)} entries ({n_articles} articles, {n_sections} sections)")

    # ── Phase 3: Match body lines to TOC entries and promote ────────
    promoted = 0
    new_lines = []
    skip_next_title = False

    for i, line in enumerate(lines):
        # Pass through TOC zone unchanged
        if toc_start <= i <= toc_end:
            new_lines.append(line)
            continue

        stripped = line.strip()

        # Handle split ARTICLE: skip the title line that follows "# ARTICLE X"
        if skip_next_title:
            if not stripped:
                new_lines.append(line)  # keep blank lines
                continue
            # Check if this line is the ARTICLE title continuation
            clean_check = re.sub(r'^#{1,6}\s+', '', stripped)
            clean_check = _strip_markup(clean_check)
            # If it's not a numbered section, it's likely the split title — skip it
            if clean_check and not re.match(r'^\d+\.\d+', clean_check):
                skip_next_title = False
                continue
            else:
                skip_next_title = False
                # Fall through to normal processing

        clean = re.sub(r'^#{1,6}\s+', '', stripped)
        clean_plain = _strip_markup(clean)
        clean_norm = _normalize(clean_plain)

        matched = False

        # Try full TOC title match
        for toc_title, level in toc_entries:
            toc_norm = _normalize(toc_title)
            if clean_norm.startswith(toc_norm):
                heading_prefix = '#' if level == 'article' else '##'
                new_lines.append(f'{heading_prefix} {toc_title}')

                remainder = clean_norm[len(toc_norm):].strip().lstrip('.').strip()
                if remainder:
                    body_text = _extract_body_after_title(clean, toc_title)
                    if body_text:
                        new_lines.append('')
                        new_lines.append(body_text)

                promoted += 1
                matched = True
                break

        if not matched:
            # Check for split ARTICLE heading (e.g. "# ARTICLE 1" alone)
            article_key = _normalize(re.sub(r'^#{1,6}\s+', '', stripped))
            article_key = re.sub(r'\.\s*$', '', article_key).upper()

            if article_key in article_lookup:
                full_title = article_lookup[article_key]
                new_lines.append(f'# {full_title}')
                skip_next_title = True
                promoted += 1
                matched = True

        if not matched:
            # Demote false LlamaParse headings like "# (a)" or "### (i) Title"
            if re.match(r'^#{1,6}\s*\([a-zA-Z0-9ivxlc]+\)', stripped, re.IGNORECASE):
                new_lines.append(clean)
            else:
                new_lines.append(line)

    if promoted:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        logging.info(f"  [HEADINGS] Promoted {promoted} body sections using TOC reference.")


def _promote_headings_basic(md_path, lines):
    """Fallback: basic ARTICLE/SECTION regex promotion when no TOC exists."""
    promoted = 0
    new_lines = []

    for line in lines:
        stripped = line.strip()
        clean = re.sub(r'^#{1,6}\s+', '', stripped)

        if re.match(r'^ARTICLE\s+([IVXLCDM]+|\d+)\.?\s*$', clean, re.IGNORECASE):
            new_lines.append(f'# {clean}')
            promoted += 1
        elif re.match(r'^SECTION\s+\d+[\./\d]*\.?\s+\S', clean, re.IGNORECASE):
            new_lines.append(f'## {clean}')
            promoted += 1
        elif re.match(r'^\d+\.\d+[\./\d]*\.?\s+\S', clean):
            # Bare numbered section (e.g. "8.01 Events of Default")
            new_lines.append(f'## {clean}')
            promoted += 1
        else:
            # Demote false LlamaParse headings like "# (a)" or "### (i) Title"
            if re.match(r'^#{1,6}\s*\([a-zA-Z0-9ivxlc]+\)', stripped, re.IGNORECASE):
                new_lines.append(clean)
            else:
                new_lines.append(line)

    if promoted:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        logging.info(f"  [HEADINGS] Promoted {promoted} legal patterns to ATX headings (basic mode).")


# ── Unified Extraction Dispatcher ───────────────────────────────────────

def extract_to_md(file_path, output_dir):
    """
    Unified dispatcher: auto-detect format by extension and call the right converter.
    Returns the path to the output .md file, or None on failure.
    For .md files, copies to output_dir if not already there.
    """
    if not os.path.isfile(file_path):
        logging.error("Input file not found: %s", file_path)
        return None

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        was_cached = os.path.exists(
            os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.md")
        )
        result = extract_pdf(file_path, output_dir)
        if result and not was_cached:
            # Only promote headings on freshly extracted files
            _promote_headings(result)
        return result

    elif ext == ".md":
        # Pass-through: copy to output_dir if needed
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, base_name)
        try:
            if os.path.abspath(file_path) != os.path.abspath(output_path):
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy2(file_path, output_path)
                logging.info(f"  -> Copied MD: {output_path}")
            return output_path
        except OSError as exc:
            logging.error("Failed to copy Markdown input %s: %s", file_path, exc)
            return None

    else:
        logging.error(f"Unsupported file format: {ext}")
        return None
