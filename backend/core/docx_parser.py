import os
import re
import json
import uuid
import socket
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from docx import Document
from docx.oxml.ns import qn
from backend.storage.minio_client import minio_client
from backend.config.config import config

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
    
    def save_image_to_minio(self, image_blob: bytes, image_index: int, section: str = "", doc_id: str = "") -> str:
        safe_section = re.sub(r'[^\w\u4e00-\u9fff]+', '_', section) if section else "unknown"
        if len(safe_section) > 100:
            safe_section = safe_section[:100]
        filename = f"images/{doc_id}/img_{safe_section}_{image_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}.png"
        
        minio_client.upload_file(filename, image_blob)
        return filename
    
    def save_image_to_local(self, image_blob: bytes, image_index: int, section: str = "", doc_id: str = "") -> Tuple[str, str]:
        safe_section = re.sub(r'[^\w\u4e00-\u9fff]+', '_', section) if section else "unknown"
        if len(safe_section) > 100:
            safe_section = safe_section[:100]
        filename = f"img_{safe_section}_{image_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}.png"
        
        local_dir = os.path.join(config.IMAGE_STORAGE_DIR, doc_id)
        os.makedirs(local_dir, exist_ok=True)
        
        local_path = os.path.join(local_dir, filename)
        with open(local_path, 'wb') as f:
            f.write(image_blob)
        
        relative_url = f"{config.IMAGE_URL_PREFIX}/{doc_id}/{filename}"

        # 优先使用 SERVER_IP（局域网可达），否则兜底
        host = config.SERVER_IP or config.SERVER_HOST
        if not host or host == "0.0.0.0":
            try:
                host = socket.gethostbyname(socket.gethostname())
            except Exception:
                host = "127.0.0.1"

        http_url = f"http://{host}:{config.SERVER_PORT}{relative_url}"

        return local_path, http_url
    
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
        patterns = [
            r'图\s*(\d+(?:[.-]\d+)?)\s*(.+)$',
            r'Figure\s+(\d+(?:[.-]\d+)?)\s*(.+)$',
            r'表\s*(\d+(?:[.-]\d+)?)\s*(.+)$',
            r'Table\s+(\d+(?:[.-]\d+)?)\s*(.+)$'
        ]
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return match.group(2) if len(match.groups()) > 1 else text
        return text if len(text) < 100 else ""

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
    def __init__(self, doc_path: str, doc_id: str):
        self.doc_path = doc_path
        self.doc_id = doc_id
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
    
    def _extract_images_and_tables(self, blocks: Dict) -> Dict:
        doc = self.parser.doc
        all_paragraphs = list(doc.paragraphs)
        
        block_ranges = {}
        for block_key, block_info in blocks.items():
            block_ranges[block_key] = (block_info['start_para'], block_info['end_para'])
        
        image_parts = {}
        for rel_id, rel in doc.part.related_parts.items():
            if 'image' in rel.content_type:
                image_parts[rel_id] = rel
        
        for block_key in blocks:
            blocks[block_key]['images'] = []
            blocks[block_key]['tables'] = []
        
        image_idx = 0
        for para_idx, para in enumerate(all_paragraphs):
            current_block = None
            for block_key, (start, end) in block_ranges.items():
                if start <= para_idx < end:
                    current_block = block_key
                    break
            if not current_block:
                continue
            
            for run in para.runs:
                blips = run.element.xpath('.//a:blip')
                for blip in blips:
                    rel_id = blip.get(qn('r:embed'))
                    if rel_id and rel_id in image_parts:
                        image_idx += 1
                        img_title = self._find_image_caption_below(para_idx, all_paragraphs, block_ranges, current_block)
                        image_blob = image_parts[rel_id].blob
                        
                        minio_path = ""
                        local_path = ""
                        http_url = ""
                        
                        try:
                            minio_path = self.parser.save_image_to_minio(
                                image_blob, image_idx, current_block, self.doc_id
                            )
                        except Exception as e:
                            print(f"[WARNING] 保存图片到MinIO失败: {e}")
                        
                        try:
                            local_path, http_url = self.parser.save_image_to_local(
                                image_blob, image_idx, current_block, self.doc_id
                            )
                        except Exception as e:
                            print(f"[WARNING] 保存图片到本地失败: {e}")
                        
                        blocks[current_block]['images'].append({
                            'index': image_idx,
                            'title': img_title,
                            'minio_path': minio_path,
                            'local_path': local_path,
                            'image_url': http_url
                        })
        
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