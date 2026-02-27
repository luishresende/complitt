import re
import os

from markdownify import markdownify as md

_last_word_re  = re.compile(r"\b\w+\b(?=\W*$)", flags=re.UNICODE)  # última palavra antes do fim do bloco
_first_word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
_div_pattern = re.compile(r'<div[^>]*>.*?</div>', flags=re.IGNORECASE | re.DOTALL)
_starts_with_lower = re.compile(r"^\W*[a-záàâãéêíóôõúç]", re.UNICODE)
_ends_with_hyphen = re.compile(r"-\s*$")
_ends_with_lower_or_comma = re.compile(r"[a-záàâãéêíóôõúç,]\s*$", re.UNICODE)
_ends_with_sentence_punctuation = re.compile(
    r'(?:[:.!?…]+["\'”’]?|["”])$'
)
_is_heading = re.compile(r"^\s*#{1,6}\s+")

# Pattern para capturar tabelas HTML
_table_pattern = re.compile(
    r'<table[^>]*>.*?</table>',
    flags=re.DOTALL | re.IGNORECASE
)


def adjust_text(md_content):
    # split por qualquer número de quebras de linha
    blocks = re.split(r'\n+', md_content)

    parts = []
    last_word = None
    prev_block_is_heading = False

    for block in blocks:
        block = block.strip()

        continued = False
        prev_block = parts[-1] if parts else None

        # primeira palavra do bloco atual
        first_match = _first_word_re.search(block)
        first_word = first_match.group(0) if first_match else None
        block_is_heading = _is_heading.match(block)

        # Regra 1: palavra duplicada que "une" dois blocos
        if last_word and first_word and last_word.lower() == first_word.lower():
            first_word_pos = block.find(first_word)
            # remove a duplicata do bloco anterior
            if prev_block in block[0:first_word_pos + len(first_word)]:
                parts.pop(-1)
        # Regra 2: hifenização OCR
        elif (
                parts
                and prev_block
                and not (block_is_heading or prev_block_is_heading)
                and _ends_with_hyphen.search(prev_block)
                and _starts_with_lower.search(block)
        ):
            # remove o hífen final do bloco anterior
            parts[-1] = _ends_with_hyphen.sub("", parts[-1])
            continued = True

        # Regra 3: NÃO termina com pontuação de final de frase
        elif (
                prev_block
                and not _ends_with_sentence_punctuation.search(prev_block)
                and not (block_is_heading or prev_block_is_heading)
                and prev_block != "[IMAGE]" and block != "[IMAGE]"
            ):
            if not (prev_block.endswith(" ") or block.endswith(" ")):
                parts.append(" ")
            continued = True

        # separador entre blocos, se não for continuidade
        if parts and not continued:
            parts.append("\n\n")

        parts.append(block)

        # atualiza last_word com a última palavra do bloco atual
        last_match = _last_word_re.search(block)
        last_word = last_match.group(0) if last_match else None
        prev_block_is_heading = block_is_heading

    return ''.join(parts)


def replace_divs(text):
    """
    Substitui cada <div> pelo valor do alt em uppercase.
    O regex captura a div inteira sem se importar com o conteúdo.
    """

    # Função de substituição
    def replacer(match):
        div_content = match.group(0)
        # Find "alt" in div
        alt_match = re.search(r'alt="([^"]+)"', div_content, flags=re.IGNORECASE)
        if alt_match:
            return f"[{alt_match.group(1).upper()}]"
        else:
            return ''  # ou '[IMAGE]' se quiser marcar div sem alt

    return _div_pattern.sub(replacer, text)


def replace_tables(text):
    """
    Substitui cada <table> HTML por sua versão em Markdown.
    """

    def replacer(match):
        table_html = match.group(0)
        # Converte a tabela HTML para Markdown
        markdown_table = md(table_html, strip=['style', 'border'])
        return markdown_table.strip()

    return _table_pattern.sub(replacer, text)


def extract_id(filename):
    match = re.search(r"_(\d+)\.md$", filename)
    if not match:
        raise ValueError(f"Nome inválido: {filename}")
    return int(match.group(1))


def merge_md_files(md_files):
    parts = []
    for file in md_files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

            if content:
                parts.append(content)

    return "\n\n".join(parts)


def get_md_files(input_path):
    files = [
        os.path.join(input_path, f) for f in os.listdir(input_path)
        if f.lower().endswith(".md") and not f.startswith("merged")
    ]

    files = sorted(files, key=extract_id)
    return files


def fix_content_pipeline(md_content, *preprocess):
    for process in preprocess:
        md_content = process(md_content)
    return md_content
