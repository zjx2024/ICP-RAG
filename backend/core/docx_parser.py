import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple
from docx import Document
from docx.oxml.ns import qn
from backend.core.parsers.records import source_file_name
from backend.storage.artifact_store import artifact_store
from backend.config.config import config


_IMAGE_SUFFIX_BY_CONTENT_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
    "image/x-emf": ".emf",
    "image/emf": ".emf",
    "image/x-wmf": ".wmf",
    "image/wmf": ".wmf",
}
_SAFE_IMAGE_SUFFIXES = frozenset(_IMAGE_SUFFIX_BY_CONTENT_TYPE.values())

class DocxParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = None
    
    def load(self) -> bool:
        try:
            self.doc = Document(self.file_path)
            return True
        except Exception:
            return False
    
    def save_image_artifact(
        self,
        image_blob: bytes,
        image_index: int,
        section: str = "",
        doc_id: str = "",
        *,
        content_type: str = "",
        source_name: str = "",
    ) -> Tuple[str, str]:
        """Persist one evidence image and return its key and public URL.

        ``images`` is the only public ArtifactStore namespace.  The local path
        points to that same physical artifact, so the old MinIO + LocalFS
        double-write is eliminated.
        """
        safe_section = re.sub(r'[^\w\u4e00-\u9fff]+', '_', section) if section else "unknown"
        if len(safe_section) > 100:
            safe_section = safe_section[:100]
        declared_type = str(content_type or "").split(";", 1)[0].strip().lower()
        source_suffix = os.path.splitext(str(source_name or ""))[1].lower()
        suffix = _IMAGE_SUFFIX_BY_CONTENT_TYPE.get(declared_type)
        if not suffix and source_suffix in _SAFE_IMAGE_SUFFIXES:
            suffix = source_suffix
        if not suffix:
            # Preserve unknown Word drawing bytes without falsely labelling
            # them as PNG. Vision parsing can then fail visibly while the
            # original engineering evidence remains downloadable.
            suffix = ".bin"
        artifact_content_type = declared_type or "application/octet-stream"
        content_id = hashlib.sha256(image_blob).hexdigest()[:16]
        filename = f"img_{safe_section}_{image_index}_{content_id}{suffix}"
        storage_key = f"images/{doc_id}/{filename}"
        artifact_store.upload_file(
            storage_key,
            image_blob,
            content_type=artifact_content_type,
            original_name=filename,
        )

        relative_url = f"{config.IMAGE_URL_PREFIX}/{doc_id}/{filename}"
        # Consumers obtain a scoped local path through materialize(storage_key).
        return storage_key, relative_url
    
    def parse_table_to_markdown(self, table) -> str:
        rows = []
        for row in table.rows:
            row_cells = [cell.text.strip().replace('\n', ' ').replace('|', '\\|') for cell in row.cells]
            rows.append(row_cells)
        if not rows:
            return ""
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")
        lines = ['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---'] * max_cols) + ' |']
        for row in rows[1:]:
            lines.append('| ' + ' | '.join(row) + ' |')
        return '\n'.join(lines)
    
    def parse_table_to_json(self, table) -> Dict:
        data = [[cell.text.strip().replace('\n', ' ') for cell in row.cells] for row in table.rows]
        return {"rows": len(data), "columns": len(data[0]) if data else 0, "data": data}
    
    def is_heading_paragraph(self, paragraph) -> Tuple[bool, int, str]:
        if paragraph.style is None:
            return False, 0, ""
        
        style_name = paragraph.style.name.lower() if paragraph.style.name else ""
        
        if 'heading' in style_name:
            match = re.search(r'heading\s*(\d+)', style_name)
            if match:
                level = int(match.group(1))
                return True, level, paragraph.style.name
        
        if '标题' in style_name:
            match = re.search(r'标题\s*(\d+)', style_name)
            if match:
                level = int(match.group(1))
                return True, level, paragraph.style.name
        
        if style_name in ['title', 'subtitle']:
            return True, 1, paragraph.style.name
        
        return False, 0, ""
    
    def extract_title_from_text(self, text: str, img_or_table: str = "image") -> str:
        if not text:
            return ""
        patterns = (
            [
                r'图\s*(\d+(?:[.-]\d+)?)\s*(.+)$',
                r'(?:Figure|Fig\.)\s+(\d+(?:[.-]\d+)?)\s*(.+)$',
            ]
            if img_or_table == "image"
            else [
                r'表\s*(\d+(?:[.-]\d+)?)\s*(.+)$',
                r'Table\s+(\d+(?:[.-]\d+)?)\s*(.+)$',
            ]
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return match.group(2) if len(match.groups()) > 1 else text
        marker_pattern = r"^(?:图|Figure|Fig\.)\s*" if img_or_table == "image" else r"^(?:表|Table)\s*"
        return text if len(text) < 100 and re.match(marker_pattern, text, re.IGNORECASE) else ""

class TextParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> bool:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            return True
        except Exception:
            return False
    
    def parse_markdown_headers(self) -> Dict:
        lines = self.content.split('\n')
        blocks = {}
        current_block = None
        current_content = []
        block_counter = 1
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                if current_block:
                    blocks[current_block['key']] = {
                        'block_id': current_block['key'],
                        'title': current_block['title'],
                        'level': current_block['level'],
                        'text': '\n'.join(current_content),
                        'images': [],
                        'tables': []
                    }
                
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                safe_title = re.sub(r'[^\w\u4e00-\u9fff]+', '_', title[:30])
                block_key = f"h{level}_{block_counter:03d}_{safe_title}"
                
                current_block = {
                    'key': block_key,
                    'title': title,
                    'level': level
                }
                current_content = []
                block_counter += 1
            else:
                if line.strip():
                    current_content.append(line.strip())
        
        if current_block:
            blocks[current_block['key']] = {
                'block_id': current_block['key'],
                'title': current_block['title'],
                'level': current_block['level'],
                'text': '\n'.join(current_content),
                'images': [],
                'tables': []
            }
        
        if not blocks:
            blocks['fallback_001'] = {
                'block_id': 'fallback_001',
                'title': '文档内容',
                'level': 1,
                'text': self.content,
                'images': [],
                'tables': []
            }
        
        return blocks

class DocumentParser:
    def __init__(
        self, doc_path: str, doc_id: str, *, source_name: Optional[str] = None
    ):
        self.doc_path = doc_path
        self.doc_id = doc_id
        self.source_name = source_file_name(doc_path, source_name)
        self.file_ext = os.path.splitext(doc_path)[1].lower()
        self.parser = None
        
        if self.file_ext == '.docx':
            self.parser = DocxParser(doc_path)
        else:
            self.parser = TextParser(doc_path)
    
    def process(self) -> Dict:
        if self.file_ext == '.docx':
            if not self.parser.load():
                return {"error": "文档加载失败"}
            blocks = self._extract_blocks_by_heading_style()
            blocks = self._extract_images_and_tables(blocks)
            return {
                "doc_id": self.doc_id,
                "blocks": blocks,
                "total_blocks": len(blocks)
            }
        else:
            from .unified_parser import UnifiedParser
            unified_parser = UnifiedParser(self.doc_path, self.doc_id)
            return unified_parser.process()
    
    def _extract_blocks_by_heading_style(self) -> Dict:
        all_paragraphs = list(self.parser.doc.paragraphs)
        
        headings = []
        for idx, para in enumerate(all_paragraphs):
            is_heading, level, style_name = self.parser.is_heading_paragraph(para)
            text = para.text.strip()
            if is_heading and text:
                headings.append({
                    'index': idx,
                    'level': level,
                    'text': text,
                    'style_name': style_name,
                    'paragraph': para
                })
        
        if not headings:
            return self._fallback_extract(all_paragraphs)
        
        def has_content(heading_info, headings_list, all_paras):
            current_idx = headings_list.index(heading_info)
            start_idx = heading_info['index'] + 1
            next_idx = headings_list[current_idx + 1]['index'] if current_idx + 1 < len(headings_list) else len(all_paras)
            
            for j in range(start_idx, next_idx):
                para_text = all_paras[j].text.strip()
                if para_text:
                    is_heading, _, _ = self.parser.is_heading_paragraph(all_paras[j])
                    if not is_heading:
                        return True
            return False
        
        blocks = {}
        i = 0
        block_counter = 1
        
        while i < len(headings):
            current = headings[i]
            current_has_content = has_content(current, headings, all_paragraphs)
            
            safe_title = re.sub(r'[^\w\u4e00-\u9fff]+', '_', current['text'][:30])
            block_key = f"h{current['level']}_{block_counter:03d}_{safe_title}"
            
            if current_has_content:
                start_idx = current['index']
                next_idx = headings[i + 1]['index'] if i + 1 < len(headings) else len(all_paragraphs)
                
                content_lines = []
                for j in range(start_idx + 1, next_idx):
                    para_text = all_paragraphs[j].text.strip()
                    if para_text:
                        is_heading, _, _ = self.parser.is_heading_paragraph(all_paragraphs[j])
                        if not is_heading:
                            content_lines.append(para_text)
                
                blocks[block_key] = {
                    'block_id': block_key,
                    'title': current['text'],
                    'level': current['level'],
                    'style_name': current['style_name'],
                    'text': '\n'.join(content_lines) if content_lines else "",
                    'start_para': start_idx,
                    'end_para': next_idx,
                    'images': [],
                    'tables': []
                }
                i += 1
                block_counter += 1
            
            else:
                if i + 1 >= len(headings):
                    blocks[block_key] = {
                        'block_id': block_key,
                        'title': current['text'],
                        'level': current['level'],
                        'style_name': current['style_name'],
                        'text': "",
                        'start_para': current['index'],
                        'end_para': current['index'] + 1,
                        'images': [],
                        'tables': []
                    }
                    i += 1
                    block_counter += 1
                else:
                    next_heading = headings[i + 1]
                    start_idx = next_heading['index']
                    next_next_idx = headings[i + 2]['index'] if i + 2 < len(headings) else len(all_paragraphs)
                    
                    content_lines = [next_heading['text']]
                    
                    for j in range(start_idx + 1, next_next_idx):
                        para_text = all_paragraphs[j].text.strip()
                        if para_text:
                            is_heading, _, _ = self.parser.is_heading_paragraph(all_paragraphs[j])
                            if not is_heading:
                                content_lines.append(para_text)
                    
                    blocks[block_key] = {
                        'block_id': block_key,
                        'title': current['text'],
                        'level': current['level'],
                        'style_name': current['style_name'],
                        'text': '\n'.join(content_lines) if content_lines else "",
                        'start_para': current['index'],
                        'end_para': next_next_idx,
                        'images': [],
                        'tables': [],
                        'merged_with': next_heading['text']
                    }
                    i += 2
                    block_counter += 1
        
        return blocks
    
    def _fallback_extract(self, all_paragraphs) -> Dict:
        blocks = {}
        block_counter = 1
        current_content = []
        
        for para in all_paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if len(text) < 100 and text[0].isdigit() and re.match(r'^(\d+(?:\.\d+)*)\s+', text):
                if current_content:
                    block_key = f"fallback_{block_counter:03d}"
                    blocks[block_key] = {
                        'block_id': block_key,
                        'title': f"段落块{block_counter}",
                        'level': 1,
                        'text': '\n'.join(current_content),
                        'images': [],
                        'tables': []
                    }
                    block_counter += 1
                    current_content = []
            else:
                current_content.append(text)
        
        if current_content:
            block_key = f"fallback_{block_counter:03d}"
            blocks[block_key] = {
                'block_id': block_key,
                'title': f"段落块{block_counter}",
                'level': 1,
                'text': '\n'.join(current_content),
                'images': [],
                'tables': []
            }
        return blocks
    
    @staticmethod
    def _image_relationship_ids(element) -> List[str]:
        """Return DrawingML and legacy VML image relationships in source order."""

        result: List[str] = []
        try:
            for blip in element.xpath('.//a:blip'):
                rel_id = blip.get(qn('r:embed')) or blip.get(qn('r:link'))
                if rel_id:
                    result.append(rel_id)
        except Exception:
            pass
        try:
            # lxml's bare ``xpath`` namespace map is not guaranteed to expose
            # the legacy VML prefix.  Expanded tags work for both normal VML
            # drawings and VML nested inside text boxes.
            for image in element.iter('{urn:schemas-microsoft-com:vml}imagedata'):
                rel_id = image.get(qn('r:id'))
                if rel_id:
                    result.append(rel_id)
        except Exception:
            pass
        return result

    def _extract_images_and_tables(self, blocks: Dict) -> Dict:
        doc = self.parser.doc
        all_paragraphs = list(doc.paragraphs)
        
        block_ranges = {}
        for block_key, block_info in blocks.items():
            block_ranges[block_key] = (
                int(block_info.get('start_para', 0)),
                int(block_info.get('end_para', len(all_paragraphs))),
            )
        
        image_parts = {}
        for rel_id, rel in doc.part.related_parts.items():
            if 'image' in rel.content_type:
                image_parts[rel_id] = rel
        
        for block_key in blocks:
            blocks[block_key]['images'] = []
            blocks[block_key]['tables'] = []
        
        body = doc.element.body
        
        heading_elem_indices = []
        for block_key, (start_para, end_para) in block_ranges.items():
            if start_para < len(all_paragraphs):
                para_elem = all_paragraphs[start_para]._element
                for idx, elem in enumerate(body):
                    if elem is para_elem:
                        heading_elem_indices.append((idx, block_key, all_paragraphs[start_para].text[:50]))
                        break
        
        heading_elem_indices.sort(key=lambda x: x[0])

        def block_for_body_index(body_index: int) -> Optional[str]:
            target = None
            for element_index, block_key, _title in heading_elem_indices:
                if element_index <= body_index:
                    target = block_key
                else:
                    break
            if target:
                return target
            return next(iter(blocks), None)

        # Walk body descendants rather than only ``doc.paragraphs``/runs.  This
        # includes images inside table cells, nested tables, text boxes and
        # legacy VML containers while retaining a stable body/relationship
        # locator. Header/footer parts remain outside the document body and are
        # deliberately not promoted without a separate scope model.
        paragraph_index_by_element = {
            paragraph._element: index for index, paragraph in enumerate(all_paragraphs)
        }
        image_idx = 0
        for body_index, element in enumerate(body):
            current_block = block_for_body_index(body_index)
            if not current_block:
                continue
            rel_ids = self._image_relationship_ids(element)
            for occurrence, rel_id in enumerate(rel_ids, start=1):
                image_part = image_parts.get(rel_id)
                if image_part is None:
                    continue
                image_blob = getattr(image_part, 'blob', b'') or b''
                if not image_blob:
                    continue
                image_idx += 1
                paragraph_index = paragraph_index_by_element.get(element)
                img_title = (
                    self._find_image_caption_below(
                        paragraph_index, all_paragraphs, block_ranges, current_block
                    )
                    if paragraph_index is not None
                    else ""
                )
                declared_type = str(getattr(image_part, "content_type", "") or "")
                source_name = str(getattr(image_part, "partname", "") or "")
                try:
                    storage_key, image_url = self.parser.save_image_artifact(
                        image_blob,
                        image_idx,
                        current_block,
                        self.doc_id,
                        content_type=declared_type,
                        source_name=source_name,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"保存文档证据图失败（image={image_idx}）: {exc}"
                    ) from exc
                tag = str(getattr(element, 'tag', '') or '')
                container = 'table' if tag.endswith('tbl') else (
                    'paragraph' if tag.endswith('p') else 'body_object'
                )
                source_ref = (
                    f"{self.source_name}#body={body_index + 1};"
                    f"container={container};rel={rel_id};occurrence={occurrence}"
                )
                blocks[current_block]['images'].append({
                    'index': image_idx,
                    'title': img_title,
                    'storage_key': storage_key,
                    # Deprecated compatibility alias for the physical DB column.
                    'minio_path': storage_key,
                    'local_path': "",
                    'image_url': image_url,
                    'content_type': declared_type or 'application/octet-stream',
                    'source_name': source_name,
                    'source_ref': source_ref,
                    'container': container,
                })
        
        table_idx = 0
        for table in doc.tables:
            table_idx += 1
            table_element = table._element
            
            table_elem_index = -1
            for idx, elem in enumerate(body):
                if elem is table_element:
                    table_elem_index = idx
                    break
            
            if table_elem_index == -1:
                target_block = self._find_table_block_by_distance(table_idx, block_ranges, all_paragraphs)
            else:
                target_block = None
                for i in range(len(heading_elem_indices) - 1, -1, -1):
                    elem_idx, block_key, _ = heading_elem_indices[i]
                    if elem_idx < table_elem_index:
                        target_block = block_key
                        break
                if not target_block and heading_elem_indices:
                    target_block = heading_elem_indices[0][1]
            
            if target_block and target_block in blocks:
                tbl_title = self._find_table_caption_above(table, target_block, block_ranges, all_paragraphs, doc)
                table_markdown = self.parser.parse_table_to_markdown(table)
                table_json = self.parser.parse_table_to_json(table)
                
                blocks[target_block]['tables'].append({
                    'index': table_idx,
                    'title': tbl_title,
                    'markdown': table_markdown,
                    'meta_data': table_json
                })
        
        return blocks
    
    def _find_image_caption_below(self, image_para_idx: int, all_paragraphs: List, block_ranges: Dict, current_block: str) -> str:
        _, block_end = block_ranges.get(current_block, (0, len(all_paragraphs)))
        
        for check_idx in range(image_para_idx + 1, min(block_end, image_para_idx + 5)):
            if check_idx >= len(all_paragraphs):
                break
            
            para_text = all_paragraphs[check_idx].text.strip()
            if not para_text:
                continue
            
            is_heading, _, _ = self.parser.is_heading_paragraph(all_paragraphs[check_idx])
            if is_heading:
                break
            
            title = self.parser.extract_title_from_text(para_text, "image")
            if title:
                return title
            
            if len(para_text) < 100 and ('图' in para_text or 'Figure' in para_text or 'fig.' in para_text.lower()):
                return para_text
        
        return ""
    
    def _find_table_caption_above(self, table, target_block: str, block_ranges: Dict, all_paragraphs: List, doc) -> str:
        if target_block not in block_ranges:
            return ""
        
        start_para, end_para = block_ranges[target_block]
        
        table_element = table._element
        body = doc.element.body
        
        table_elem_index = -1
        for idx, elem in enumerate(body):
            if elem is table_element:
                table_elem_index = idx
                break
        
        if table_elem_index >= 0:
            for idx in range(table_elem_index - 1, max(table_elem_index - 5, -1), -1):
                elem = body[idx]
                if elem.tag.endswith('p'):
                    for para in all_paragraphs:
                        if para._element is elem:
                            para_text = para.text.strip()
                            if para_text:
                                is_heading, _, _ = self.parser.is_heading_paragraph(para)
                                if is_heading:
                                    return ""
                                
                                title = self.parser.extract_title_from_text(para_text, "table")
                                if title:
                                    return title
                                
                                if len(para_text) < 100 and ('表' in para_text or 'Table' in para_text):
                                    return para_text
                            break
        
        for para_idx in range(min(end_para, len(all_paragraphs)) - 1, max(start_para, -1), -1):
            para = all_paragraphs[para_idx]
            para_text = para.text.strip()
            if not para_text:
                continue
            
            is_heading, _, _ = self.parser.is_heading_paragraph(para)
            if is_heading:
                return ""
            
            title = self.parser.extract_title_from_text(para_text, "table")
            if title:
                return title
            
            if len(para_text) < 100 and ('表' in para_text or 'Table' in para_text):
                return para_text
        
        return ""
    
    def _find_table_block_by_distance(self, table_idx: int, block_ranges: Dict, all_paragraphs: List) -> Optional[str]:
        block_starts = [(block_key, info['start_para']) for block_key, info in block_ranges.items()]
        block_starts.sort(key=lambda x: x[1])
        
        target_block = None
        for block_key, start_para in block_starts:
            if start_para <= table_idx:
                target_block = block_key
            else:
                break
        return target_block
