#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core BibTeX cleaning/formatting logic without GUI deps.
Extracted from process_bib.py (v7.2) for reuse (e.g., Pyodide/web).
"""

import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict, List, Tuple, Any
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / 'data'


def load_terms_from_file(filename: str, default: Set[str]) -> Set[str]:
    """
    Load newline-delimited terms from data/filename, ignoring blank lines and comments.
    Returns the provided default set if the file is missing or unreadable.
    """
    path = DATA_DIR / filename
    try:
        with path.open('r', encoding='utf-8') as handle:
            terms = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith('#')
            ]
        if terms:
            return set(terms)
    except FileNotFoundError:
        pass
    except OSError:
        pass
    return set(default)

def parse_simple_yaml_list(path: Path, default: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Minimal YAML-ish parser to avoid external deps.
    Supports a list of maps with optional list fields (e.g., variants).
    Falls back to default if parsing fails or file missing.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return list(default)
    except OSError:
        return list(default)

    entries: List[Dict[str, Any]] = []
    current: Dict[str, Any] = None
    current_list_key: str = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if stripped.startswith('- '):
            payload = stripped[2:].lstrip()
            # Treat as a new entry when the line defines a field (e.g., "- id: cvpr")
            if payload.startswith('id'):
                if current:
                    entries.append(current)
                current = {}
                current_list_key = None
                if ':' in payload:
                    key, val = payload.split(':', 1)
                    key = key.strip()
                    val = val.strip().strip('\'"')
                    if key:
                        current[key] = val
                continue
            # Otherwise, if we're inside a list, treat it as a list item
            if current is not None and current_list_key:
                val = payload.strip().strip('\'"')
                if val:
                    current[current_list_key].append(val)
                continue

        if current is None:
            continue

        # Key/value or list start
        if ':' in stripped:
            key, val = stripped.split(':', 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if val == '':
                current_list_key = key
                current[current_list_key] = []
            else:
                current[key] = val.strip('\'"')
                current_list_key = None
            continue

        # Fallback: treat as list continuation
        if current_list_key:
            current[current_list_key].append(stripped)

    if current:
        entries.append(current)

    return entries if entries else list(default)


def load_venue_entries() -> List[Dict[str, Any]]:
    """
    Load venue definitions from data/venues/*.yml if present; fallback to legacy venues.yml.
    Combines all YAML lists into a single list in deterministic order.
    """
    entries: List[Dict[str, Any]] = []
    venue_dir = DATA_DIR / 'venues'

    try:
        files = sorted([p for p in venue_dir.glob('*.yml') if p.is_file()])
    except OSError:
        files = []

    for path in files:
        entries.extend(parse_simple_yaml_list(path, []))

    if entries:
        return entries

    legacy = parse_simple_yaml_list(DATA_DIR / 'venues.yml', [])
    return legacy if legacy else list(DEFAULT_VENUES)


def normalize_venue_key(text: str) -> str:
    """Normalize venue text for lookup (lowercase, strip punctuation/extra spaces)."""
    text = text.lower()
    # Drop ordinal/edition markers so numbered proceedings (e.g., "29th") match base venues.
    text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', ' ', text)
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compile_venue_patterns(entries: List[Dict[str, Any]]) -> List[Tuple[re.Pattern, str]]:
    """
    Compile regex patterns declared in venue entries.
    Each entry may have a 'patterns' list with regex strings (Python syntax).
    """
    compiled: List[Tuple[re.Pattern, str]] = []
    for entry in entries:
        vid = str(entry.get('id', '')).strip()
        if not vid:
            continue
        patterns = entry.get('patterns') or []
        if not isinstance(patterns, (list, tuple)):
            continue
        for pat in patterns:
            raw_pat = str(pat)
            try:
                compiled.append((re.compile(raw_pat, flags=re.IGNORECASE), vid))
            except re.error:
                continue
    return compiled


def build_venue_lookup(entries: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    Build lookup tables:
      - by_id: id -> entry dict
      - key_to_id: normalized text -> id
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    key_to_id: Dict[str, str] = {}
    for entry in entries:
        vid = str(entry.get('id', '')).strip()
        if not vid:
            continue
        by_id[vid] = entry
        candidates: List[str] = []
        for field in ['full', 'abbreviation', 'acronym']:
            val = entry.get(field)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())
        variants = entry.get('variants') or []
        if isinstance(variants, (list, tuple)):
            candidates.extend([str(v) for v in variants if str(v).strip()])
        for cand in candidates:
            norm = normalize_venue_key(cand)
            if norm:
                key_to_id.setdefault(norm, vid)
    return by_id, key_to_id


def standardize_venue_name(
    text: str,
    target: str,
    key_to_id: Dict[str, str],
    by_id: Dict[str, Dict[str, Any]],
    patterns: List[Tuple[re.Pattern, str]] = None,
) -> str:
    """
    Return standardized venue text for a given target mode (full/abbr/acronym) if known.
    """
    norm = normalize_venue_key(text)
    vid = key_to_id.get(norm)

    # Fallback to regex patterns if no direct normalized match
    if not vid and patterns:
        for regex, pid in patterns:
            try:
                if regex.search(text):
                    vid = pid
                    break
            except re.error:
                continue

    if not vid:
        return ''

    entry = by_id.get(vid) or {}
    if target == 'full':
        return entry.get('full') or ''
    if target == 'abbr':
        return entry.get('abbreviation') or entry.get('acronym') or entry.get('full') or ''
    if target == 'acronym':
        return entry.get('acronym') or entry.get('abbreviation') or entry.get('full') or ''
    return ''


# Default fields to keep (can be modified via GUI or CLI)
DEFAULT_KEEP_FIELDS = {'title', 'author', 'journal', 'year', 'booktitle'}

# All common BibTeX fields
DEFAULT_ALL_BIBTEX_FIELDS = {
    'title', 'author', 'journal', 'year', 'booktitle',
    'volume', 'number', 'pages', 'month', 'publisher',
    'address', 'editor', 'edition', 'series', 'note',
    'doi', 'isbn', 'issn', 'url', 'abstract',
    'keywords', 'organization', 'school', 'institution',
    'howpublished', 'type', 'chapter', 'crossref'
}
ALL_BIBTEX_FIELDS = load_terms_from_file('all_bibtex_fields.txt', DEFAULT_ALL_BIBTEX_FIELDS)

# Only these entry types will have fields removed
REMOVE_FIELDS_FOR_TYPES = {'article', 'inproceedings'}

DEFAULT_LOWERCASE_WORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 
    'from', 'by', 'in', 'of', 'with', 'via', 'into', 'onto', 'upon', 'as',
    'vs', 'versus'
}
LOWERCASE_WORDS = load_terms_from_file('lowercase_words.txt', DEFAULT_LOWERCASE_WORDS)

DEFAULT_PRESERVE_TERMS = {'arXiv preprint', 'arXiv', 'bioRxiv', 'medRxiv', 'ChemRxiv'}
PRESERVE_TERMS = load_terms_from_file('preserve_terms.txt', DEFAULT_PRESERVE_TERMS)
PRESERVE_TERMS_LOWER = {term.lower(): term for term in PRESERVE_TERMS}

DEFAULT_PROTECTED_TERMS = {
    # ============================================================================
    # Dimensions & Spatial
    # ============================================================================
    '1D', '2D', '3D', '4D', '5D', '6D',
    
    # ============================================================================
    # Deep Learning Models & Architectures
    # ============================================================================
    'ResNet', 'VGG', 'UNet', 'YOLO', 'ViT', 'Transformer',
    'GAN', 'VAE', 'LSTM', 'CNN', 'RNN', 'GNN', 'BiLSTM', 'GRU',
    'DNN', 'BERT', 'GPT', 'CLIP',
    
    # ============================================================================
    # Computer Vision & Graphics Methods
    # ============================================================================
    'NeRF', '3DGS', 'SLAM',
    'SIFT', 'SURF', 'ORB',
    'HOG', 'DPM', 'RCNN', 'FRCNN', 'SSD',
    'RANSAC', 'ICP', 'FLANN',
    
    # ============================================================================
    # Conferences & Journals
    # ============================================================================
    'CVPR', 'ICCV', 'ECCV', 'NeurIPS', 'ICML', 'ICLR', 'AAAI', 'IJCAI',
    'SIGGRAPH', 'ACM', 'IEEE', 'ToG', 'TPAMI', 'IJCV', 'PAMI', 'TIP', 'CVIU',
    
    # ============================================================================
    # Machine Learning Algorithms & Optimization
    # ============================================================================
    'SVM', 'KNN', 'SGD', 'Adam',
    'PCA', 'ICA', 'LDA', 'GMM', 'HMM',
    
    # ============================================================================
    # Natural Language Processing & Metrics
    # ============================================================================
    'NLP', 'Seq2Seq', 'BLEU', 'ROUGE', 'METEOR', 'CIDEr', 'SPICE',
    
    # ============================================================================
    # Datasets
    # ============================================================================
    'ImageNet', 'COCO', 'VOC', 'KITTI', 'nuScenes',
    
    # ============================================================================
    # Deep Learning Frameworks & Libraries
    # ============================================================================
    'PyTorch', 'TensorFlow', 'Keras', 'JAX', 'MXNet',
    'OpenCV', 'OpenGL', 'CUDA', 'cuDNN', 'TensorRT',
    
    # ============================================================================
    # Image & Video Metrics
    # ============================================================================
    'SSIM', 'PSNR', 'MSE', 'MAE', 'IoU', 'mAP', 'FID', 'LPIPS',
    
    # ============================================================================
    # Color Spaces & Image Formats
    # ============================================================================
    'RGB', 'RGBA', 'HSV', 'CMYK', 'YUV', 'RGBD',
    'JPEG', 'PNG', 'GIF', 'TIFF', 'BMP', 'SVG', 'PDF',
    
    # ============================================================================
    # Biology & Medicine
    # ============================================================================
    'DNA', 'RNA', 'mRNA', 'tRNA', 'rRNA', 'siRNA', 'miRNA',
    'ATP', 'GTP', 'ADP', 'GDP', 'NADH', 'FADH2', 'CoA',
    'PCR', 'qPCR', 'RT-PCR', 'ELISA', 'FACS', 'CRISPR',
    'MRI', 'CT', 'PET', 'fMRI', 'EEG', 'ECG',
    
    # ============================================================================
    # Academic Degrees
    # ============================================================================
    'PhD', 'PhD.', 'MSc', 'BSc', 'MD', 'DPhil',
    
    # ============================================================================
    # Organizations & Institutions
    # ============================================================================
    'China', 'Chinese', 'USA', 'UK', 'EU', 'NASA', 'CERN', 'MIT', 'UCLA', 'ETH',
    
    # ============================================================================
    # AI & Technology Domains
    # ============================================================================
    'AI', 'ML', 'DL', 'CV', 'IoT',
    'AR', 'VR', 'XR', 'MR',
    
    # ============================================================================
    # Wireless & Communication Technologies
    # ============================================================================
    'WiFi', 'LTE', 'VoIP', 'Bluetooth', 'NFC', 'RFID',
    'MIMO', 'OFDM', 'QAM', 'BPSK', 'QPSK',
    
    # ============================================================================
    # Hardware & Computing
    # ============================================================================
    'GPU', 'CPU', 'TPU', 'FPGA', 'ASIC',
    
    # ============================================================================
    # Sensors & Positioning
    # ============================================================================
    'GPS', 'LiDAR',
    
    # ============================================================================
    # Data Formats & Protocols
    # ============================================================================
    'JSON', 'XML', 'HTML', 'CSS', 'YAML', 'CSV', 'HDF5',
    'SQL', 'NoSQL',
    'HTTP', 'HTTPS', 'FTP', 'SSH', 'TCP', 'UDP', 'IP',
    
    # ============================================================================
    # Operating Systems & Platforms
    # ============================================================================
    'MacOS', 'iOS', 'Android', 'Linux', 'Windows', 'Ubuntu',
    
    # ============================================================================
    # Development Tools
    # ============================================================================
    'LaTeX', 'BibTeX', 'GitHub', 'GitLab',
    
    # ============================================================================
    # Computer Vision Algorithms
    # ============================================================================
    'Gaussian', 'Laplacian', 'Sobel', 'Canny', 'KD-tree',
    
    # ============================================================================
    # Currency Codes
    # ============================================================================
    'USD', 'EUR', 'GBP', 'CNY', 'JPY',
    
    # ============================================================================
    # Latin Abbreviations & Common Terms
    # ============================================================================
    'vs.', 'etc.', 'i.e.', 'e.g.', 'et al.',

    'SVD', 'FFT', 'DFT', 'DCT', 'IDFT', 'IFFT', 'IDCT', 'DTCWT', 'CWT',
}
PROTECTED_TERMS = load_terms_from_file('protected_terms.txt', DEFAULT_PROTECTED_TERMS)

PROTECTED_TERMS_LOWER = {term.lower(): term for term in PROTECTED_TERMS}

# Venue definitions for standardization (full/abbr/acronym)
DEFAULT_VENUES: List[Dict[str, Any]] = [
    {
        'id': 'cvpr',
        'type': 'conference',
        'full': 'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition',
        'abbreviation': 'Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.',
        'acronym': 'CVPR',
        'variants': [
            'IEEE/CVF Conference on Computer Vision and Pattern Recognition',
            'IEEE Conference on Computer Vision and Pattern Recognition',
            'IEEE/CVF Conf. on Computer Vision and Pattern Recognition',
            'IEEE Conf. on Computer Vision and Pattern Recognition',
            'IEEE/CVF Conference on Computer Vision and Pattern Recognition-CVPR',
        ],
    },
    {
        'id': 'iccv',
        'type': 'conference',
        'full': 'Proceedings of the IEEE/CVF International Conference on Computer Vision',
        'abbreviation': 'Proc. IEEE/CVF Int. Conf. Comput. Vis.',
        'acronym': 'ICCV',
        'variants': [
            'IEEE/CVF International Conference on Computer Vision',
            'IEEE International Conference on Computer Vision',
            'IEEE/CVF Int. Conf. on Computer Vision',
            'IEEE Int. Conf. on Computer Vision',
        ],
    },
    {
        'id': 'eccv',
        'type': 'conference',
        'full': 'Proceedings of the European Conference on Computer Vision',
        'abbreviation': 'Proc. Eur. Conf. Comput. Vis.',
        'acronym': 'ECCV',
        'variants': [
            'European Conference on Computer Vision',
            'Eur. Conf. on Computer Vision',
        ],
    },
    {
        'id': 'neurips',
        'type': 'conference',
        'full': 'Conference on Neural Information Processing Systems',
        'abbreviation': 'Adv. Neural Inf. Process. Syst.',
        'acronym': 'NeurIPS',
        'variants': [
            'Advances in Neural Information Processing Systems',
            'Neural Information Processing Systems',
            'Neural Inf. Process. Syst.',
            'NIPS',
        ],
    },
    {
        'id': 'iclr',
        'type': 'conference',
        'full': 'International Conference on Learning Representations',
        'abbreviation': 'Int. Conf. Learn. Represent.',
        'acronym': 'ICLR',
        'variants': [
            'International Conf. on Learning Representations',
            'Int. Conf. on Learning Representations',
        ],
    },
    {
        'id': 'icml',
        'type': 'conference',
        'full': 'International Conference on Machine Learning',
        'abbreviation': 'Proc. Int. Conf. Mach. Learn.',
        'acronym': 'ICML',
        'variants': [
            'Int. Conf. on Machine Learning',
            'International Conf. on Machine Learning',
        ],
    },
    {
        'id': 'aaai',
        'type': 'conference',
        'full': 'Proceedings of the AAAI Conference on Artificial Intelligence',
        'abbreviation': 'Proc. AAAI Conf. Artif. Intell.',
        'acronym': 'AAAI',
        'variants': [
            'AAAI Conference on Artificial Intelligence',
            'AAAI Conf. on Artificial Intelligence',
        ],
    },
    {
        'id': 'ijcai',
        'type': 'conference',
        'full': 'Proceedings of the International Joint Conference on Artificial Intelligence',
        'abbreviation': 'Proc. Int. Joint Conf. Artif. Intell.',
        'acronym': 'IJCAI',
        'variants': [
            'International Joint Conference on Artificial Intelligence',
            'Int. Joint Conf. on Artificial Intelligence',
        ],
    },
    {
        'id': 'tpami',
        'type': 'journal',
        'full': 'IEEE Transactions on Pattern Analysis and Machine Intelligence',
        'abbreviation': 'IEEE Trans. Pattern Anal. Mach. Intell.',
        'acronym': 'TPAMI',
        'variants': [
            'IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)',
            'Trans. Pattern Analysis and Machine Intelligence',
        ],
    },
    {
        'id': 'tog',
        'type': 'journal',
        'full': 'ACM Transactions on Graphics',
        'abbreviation': 'ACM Trans. Graph.',
        'acronym': 'ToG',
        'variants': [
            'ACM Trans. on Graphics',
            'Transactions on Graphics',
        ],
    },
]
VENUE_ENTRIES = load_venue_entries()
VENUE_BY_ID, VENUE_LOOKUP = build_venue_lookup(VENUE_ENTRIES)
VENUE_PATTERNS = compile_venue_patterns(VENUE_ENTRIES)

# Minimum Tk version required for stable GUI on macOS

# Processing Statistics Class
# ============================================================================

class ProcessingLog:
    def __init__(self):
        self.total_entries = 0
        self.modified_entries = 0
        self.changes = []
        self.duplicates = []
        self.duplicates_pre_merge = []
        self.duplicates_post_merge = []
        self.pre_merge_output = ''
        self.removed_fields_count = 0
        self.title_changes_count = 0
        self.journal_changes_count = 0
        self.booktitle_changes_count = 0
        self.colon_titles_count = 0
        self.period_removed_count = 0
        self.entries_by_type = {}
        self.title_mode = 'title'
        self.keep_fields = DEFAULT_KEEP_FIELDS.copy()
        self.venue_mode = 'format'
        self.venue_standardize_mode = 'none'
        
    def add_entry(self, entry_type: str = 'unknown'):
        self.total_entries += 1
        entry_type = entry_type.lower()
        self.entries_by_type[entry_type] = self.entries_by_type.get(entry_type, 0) + 1
    
    def add_change(self, entry_key: str, change_type: str, field: str, before: str, after: str):
        self.changes.append({
            'entry': entry_key,
            'type': change_type,
            'field': field,
            'before': before,
            'after': after
        })
        if change_type == 'field_modified':
            if field == 'title':
                self.title_changes_count += 1
            elif field == 'journal':
                self.journal_changes_count += 1
            elif field == 'booktitle':
                self.booktitle_changes_count += 1
        elif change_type == 'removed_field':
            self.removed_fields_count += 1
        elif change_type == 'merged_entry':
            # No scalar counters needed; stored in changes list
            pass
    
    def add_duplicate(self, normalized_title: str, entries: List[Dict[str, str]], group_type: str = 'title', keys: List[str] = None):
        self.duplicates.append({
            'normalized_title': normalized_title,
            'group_norm': normalized_title,
            'group_type': group_type,
            'keys': keys or [],
            'entries': entries
        })
    
    def mark_modified(self):
        self.modified_entries += 1
    
    def generate_report(self) -> str:
        report = []
        report.append("=" * 80)
        report.append("BibTeX Processing Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        report.append("SUMMARY:")
        report.append(f"  Total entries processed: {self.total_entries}")
        report.append(f"  Entries modified: {self.modified_entries}")
        report.append(f"  Entries unchanged: {self.total_entries - self.modified_entries}")
        report.append(f"  Title formatting changes: {self.title_changes_count}")
        report.append(f"  Title trailing periods removed: {self.period_removed_count}")
        report.append(f"  Journal formatting changes: {self.journal_changes_count}")
        report.append(f"  Booktitle formatting changes: {self.booktitle_changes_count}")
        report.append(f"  Fields removed: {self.removed_fields_count}")
        if self.colon_titles_count > 0:
            report.append(f"  WARNING: Titles with colons: {self.colon_titles_count}")
        if self.duplicates:
            dup_count = sum(len(item['entries']) for item in self.duplicates)
            report.append(f"  Potential duplicate entries detected: {len(self.duplicates)} group(s), {dup_count} total entries")
        report.append("")
        
        report.append("ENTRIES BY TYPE:")
        for entry_type in sorted(self.entries_by_type.keys()):
            count = self.entries_by_type[entry_type]
            in_remove_list = "Yes" if entry_type in REMOVE_FIELDS_FOR_TYPES else "No"
            report.append(f"  {entry_type}: {count} (field removal: {in_remove_list})")
        report.append("")
        
        if self.changes:
            report.append("DETAILED CHANGES:")
            report.append("-" * 80)
            
            current_entry = None
            for change in self.changes:
                if current_entry != change['entry']:
                    current_entry = change['entry']
                    report.append(f"\n[Entry: {current_entry}]")
                
                if change['type'] == 'field_modified':
                    report.append(f"  {change['field'].capitalize()} changed:")
                    report.append(f"    BEFORE: {change['before']}")
                    report.append(f"    AFTER:  {change['after']}")
                elif change['type'] == 'field_renamed':
                    report.append(f"  Renamed field {change['field']}:")
                    report.append(f"    VALUE: {change['after']}")
                elif change['type'] == 'removed_field':
                    report.append(f"  Removed field '{change['field']}':")
                    report.append(f"    {change['before']}")
                elif change['type'] == 'merged_entry':
                    report.append(f"  Merged entry '{change['field']}' with: {', '.join(change['after'])}")
        else:
            report.append("No changes were made to any entries.")
        
        if self.duplicates:
            report.append("")
            report.append("POTENTIAL DUPLICATES:")
            report.append("-" * 80)
            for group in self.duplicates:
                norm = group.get('group_norm') or group.get('normalized_title') or ''
                kind = group.get('group_type', 'title')
                label = f"{norm} ({kind})" if norm else f"({kind})"
                report.append(f"Group: {label}")
                for entry in group['entries']:
                    entry_key = entry.get('key', '(no key)')
                    entry_title = entry.get('title') or '(no title)'
                    report.append(f"  - {entry_key}: {entry_title}")
            report.append("-" * 80)
        
        report.append("")
        report.append("=" * 80)
        report.append("Configuration used:")
        report.append(f"  Entry types with field removal: {', '.join(sorted(REMOVE_FIELDS_FOR_TYPES))}")
        report.append(f"  Kept fields for those types: {', '.join(sorted(self.keep_fields))}")
        report.append(f"  Other entry types: All fields kept")
        report.append(f"  Title/Journal/Booktitle formatting: Applied to ALL entry types")
        
        title_mode_text = {
            'title': 'Title case (capitalize all major words)',
            'sentence': 'Sentence case (capitalize first word only)',
            'keep': 'Keep as-is'
        }.get(self.title_mode, self.title_mode)
        report.append(f"  Title formatting: {title_mode_text}")
        report.append(f"  Title trailing period removal: Automatic")
        
        venue_mode_text = {
            'format': 'Format (title case capitalization)',
            'keep': 'Keep as-is'
        }.get(self.venue_mode, self.venue_mode)
        report.append(f"  Venue name format: {venue_mode_text}")
        
        report.append(f"  Protected terms: {len(PROTECTED_TERMS)} terms")
        report.append(f"  Lowercase words: {len(LOWERCASE_WORDS)} words")
        report.append("=" * 80)
        
        if self.colon_titles_count > 0:
            report.append("")
            report.append("!" * 80)
            report.append("WARNING: IMPORTANT REMINDER")
            report.append(f"   {self.colon_titles_count} title(s) contain colons.")
            report.append("   The text BEFORE the colon often contains custom names/methods")
            report.append("   (e.g., '{NeRF}', '{HiDDeN}').")
            report.append("   Please manually verify these are formatted correctly!")
            report.append("!" * 80)
        
        return '\n'.join(report)

# ============================================================================
# Core Processing Functions
# ============================================================================

def is_likely_method_name(text: str) -> bool:
    """Detect if text is likely a technical term/method name"""
    text = text.strip()
    if not text:
        return False
    text_no_hyphen = text.replace('-', '').replace('_', '')
    
    # Protected terms
    if text.lower() in PROTECTED_TERMS_LOWER:
        return True
    
    # All uppercase (acronyms)
    if text.isupper() and len(text) >= 2:
        return True
    
    # Contains digits
    if re.search(r'\d', text):
        return True
    
    # Greek letters
    if re.search(r'[α-ωΑ-Ω]', text):
        return True
    
    # For non-hyphenated/underscored text
    if '-' not in text and '_' not in text:
        # 1. 大驼峰 PascalCase: 首字母大写 + 包含小写转大写
        if text[0].isupper() and re.search(r'[a-z][A-Z]', text_no_hyphen):
            return True
        
        # 2. 小驼峰 camelCase: 首字母小写 + 包含小写转大写
        if text[0].islower() and re.search(r'[a-z][A-Z]', text_no_hyphen):
            return True
        
        # 3. 多个连续大写后跟小写 (例如 XMLParser, JSONData)
        if re.search(r'[A-Z]{2,}[a-z]', text_no_hyphen):
            return True
        
        # 4. 特殊情况：i开头的大驼峰 (iPhone, iPad)
        if text.startswith(('i', 'e')) and len(text) > 1 and text[1].isupper():
            return True
    
    # Hyphenated/underscored technical terms
    if '-' in text or '_' in text:
        parts = re.split(r'[-_]', text)
        tech_parts = 0
        for p in parts:
            if p and ((p.isupper() and len(p) >= 2) or re.search(r'\d', p)):
                tech_parts += 1
        if tech_parts >= 2:
            return True
    
    return False

def normalize_title_for_duplicates(title: str) -> str:
    """
    Normalize a title for duplicate comparison:
    - ignore case
    - trim trailing spaces
    - remove connector characters (hyphens/dashes/underscores)
    """
    if not title:
        return ''
    normalized = title.rstrip().lower()
    normalized = re.sub(r'[-‐‑‒–—―_]+', '', normalized)
    return normalized

def build_duplicate_groups(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group duplicate entries when they share a normalized title or a key."""
    if not entries:
        return []
    parent = list(range(len(entries)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    title_map: Dict[str, List[int]] = defaultdict(list)
    key_map: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(entries):
        norm_title = (item.get('normalized_title') or '').strip()
        key = item.get('key')
        if norm_title:
            title_map[norm_title].append(idx)
        if key:
            key_map[key].append(idx)
    
    def link_groups(index_map: Dict[str, List[int]]):
        for idxs in index_map.values():
            if len(idxs) <= 1:
                continue
            anchor = idxs[0]
            for other in idxs[1:]:
                union(anchor, other)
    
    link_groups(title_map)
    link_groups(key_map)
    
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, item in enumerate(entries):
        root = find(idx)
        grouped[root].append(item)
    
    groups = []
    for items in grouped.values():
        if len(items) <= 1:
            continue
        ordered = sorted(items, key=lambda e: e.get('order', 0))
        titles = [i.get('normalized_title') for i in ordered if i.get('normalized_title')]
        keys = [i.get('key') for i in ordered if i.get('key')]
        primary_title = titles[0] if titles else ''
        primary_key = keys[0] if keys else ''
        group_norm = primary_title or primary_key or ''
        if titles and keys:
            group_type = 'mixed'
        elif titles:
            group_type = 'title'
        else:
            group_type = 'key'
        groups.append({
            'normalized_title': group_norm,
            'group_norm': group_norm,
            'group_type': group_type,
            'titles': titles,
            'keys': keys,
            'entries': ordered
        })
    return groups

def count_lines(text: str) -> int:
    """Count visual lines the way a textarea shows them (newline count + 1)."""
    if not text:
        return 0
    return text.count('\n') + 1

def normalize_total_line_count(text: str, target_lines: int) -> str:
    """Adjust total line count to match target by trimming/adding trailing blanks."""
    if target_lines is None or target_lines <= 0:
        return text
    lines = text.split('\n')
    if len(lines) == target_lines:
        return text
    if len(lines) > target_lines:
        return '\n'.join(lines[:target_lines])
    return text + '\n' * (target_lines - len(lines))

def pad_to_line_count(text: str, target_lines: int) -> str:
    """Pad text with trailing newlines so total line count reaches target."""
    if target_lines is None or target_lines <= 0:
        return text
    current = count_lines(text)
    if current >= target_lines:
        return text
    add = target_lines - current
    return text + '\n' * add

def remove_trailing_period(title: str) -> Tuple[str, bool]:
    """
    Remove trailing period from title if it's a regular sentence-ending period.
    
    Preserves:
    - Abbreviation periods (e.g., "Ph.D.", "U.S.A.")
    - Other punctuation (?, !)
    - Periods inside braces
    
    Returns:
        (cleaned_title, was_changed)
    """
    original = title
    
    # Strip trailing whitespace
    title = title.rstrip()
    
    # If doesn't end with period, nothing to do
    if not title.endswith('.'):
        return title, False
    
    # Check if it's likely an abbreviation period
    # Pattern: word ending with capital letter(s) followed by period
    if re.search(r'[A-Z]\.$', title):
        # Could be abbreviation like "Ph.D." or "USA."
        # Check if there's a space before it (likely abbreviation)
        if re.search(r'\s[A-Z][a-z]*\.$', title) or re.search(r'\s[A-Z]+\.$', title):
            # Likely abbreviation, keep the period
            return title, False
    
    # Check for common abbreviations at the end
    common_abbrevs = [
        r'Ph\.D\.$', r'M\.Sc\.$', r'B\.Sc\.$', r'Dr\.$', r'Prof\.$',
        r'Inc\.$', r'Ltd\.$', r'Corp\.$', r'Co\.$',
        r'vs\.$', r'etc\.$', r'i\.e\.$', r'e\.g\.$', r'et al\.$'
    ]
    for pattern in common_abbrevs:
        if re.search(pattern, title):
            return title, False
    
    # Check if period is inside braces at the end
    if title.endswith('}.'):
        return title, False
    
    # Remove the trailing period
    cleaned = title[:-1].rstrip()
    
    return cleaned, True

def protect_proper_nouns(text: str, add_braces: bool = True, title_mode: str = 'title') -> str:
    """
    Protect technical terms and proper nouns with braces.
    
    Args:
        text: Input text
        add_braces: Whether to add braces around protected terms
        title_mode: 'title', 'sentence', or 'keep'
    """
    if title_mode == 'keep':
        return text
    
    # Save existing braced content
    braced_content = {}
    placeholder_pattern = '<<<BRACED_{}>>>'
    counter = 0
    
    def save_braced(match):
        nonlocal counter
        placeholder = placeholder_pattern.format(counter)
        braced_content[placeholder] = match.group(0)
        counter += 1
        return placeholder
    
    text_with_placeholders = re.sub(r'\{[^}]+\}', save_braced, text)
    
    # Replace dashes with special placeholders
    dash_placeholder = '<<<DASH>>>'
    text_with_placeholders = re.sub(r'[-–—]', dash_placeholder, text_with_placeholders)
    
    # Split by spaces
    words = text_with_placeholders.split()
    result = []
    
    for i, word in enumerate(words):
        # Strip punctuation
        clean_word = word.strip('.,;:!?()[]{}"\'-–—')
        punctuation_before = word[:len(word) - len(word.lstrip('.,;:!?()[]{}"\'-–—'))]
        punctuation_after = word[len(clean_word) + len(punctuation_before):]
        
        if not clean_word:
            result.append(word)
            continue
        
        # ✅ 如果包含 placeholder，需要特殊处理
        if '<<<BRACED_' in clean_word:
            processed_word = clean_word
            
            # ✅ 关键修改：检查是否有 dash 分隔
            # 如果有 dash，按 dash 分割成独立单词处理
            # 如果没有 dash，整个作为一个单词处理
            
            if dash_placeholder in processed_word:
                # 有破折号，分割成多个单词
                dash_parts = processed_word.split(dash_placeholder)
                processed_dash_parts = []
                
                for dp_idx, dash_part in enumerate(dash_parts):
                    if not dash_part:
                        processed_dash_parts.append('')
                        continue
                    
                    # 每个破折号分隔的部分作为独立单词处理
                    processed_part = process_word_with_placeholders(
                        dash_part, i, dp_idx, words, title_mode, braced_content
                    )
                    processed_dash_parts.append(processed_part)
                
                processed_word = dash_placeholder.join(processed_dash_parts)
            else:
                # 没有破折号，整个作为一个单词
                processed_word = process_word_with_placeholders(
                    processed_word, i, 0, words, title_mode, braced_content
                )
            
            # 恢复 dash
            processed_word = processed_word.replace(dash_placeholder, '-')
            
            # 恢复 braced content
            for placeholder, original in braced_content.items():
                processed_word = processed_word.replace(placeholder, original)
            
            result.append(f'{punctuation_before}{processed_word}{punctuation_after}')
            continue
        
        # 处理包含 dash 但不含 placeholder 的单词
        if dash_placeholder in clean_word:
            parts = clean_word.split(dash_placeholder)
            processed_parts = []
            
            for j, part in enumerate(parts):
                if not part:
                    continue
                
                if part.lower() in PROTECTED_TERMS_LOWER:
                    protected = PROTECTED_TERMS_LOWER[part.lower()]
                    if add_braces:
                        processed_parts.append(f'{{{protected}}}')
                    else:
                        processed_parts.append(protected)
                elif is_likely_method_name(part):
                    if add_braces:
                        processed_parts.append(f'{{{part}}}')
                    else:
                        processed_parts.append(part)
                else:
                    if title_mode == 'title':
                        is_start = (i == 0 and j == 0)
                        is_after_colon = False
                        if i > 0 and ':' in words[i-1]:
                            is_after_colon = True
                        
                        if part.lower() in LOWERCASE_WORDS and not is_start and not is_after_colon:
                            processed_parts.append(part.lower())
                        else:
                            processed_parts.append(part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper())
                    elif title_mode == 'sentence':
                        is_start = (i == 0 and j == 0)
                        is_after_colon = False
                        if i > 0 and ':' in words[i-1]:
                            is_after_colon = True
                        
                        if is_start or is_after_colon:
                            processed_parts.append(part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper())
                        else:
                            processed_parts.append(part.lower())
                    else:
                        processed_parts.append(part)
            
            processed_word = '-'.join(processed_parts)
            result.append(f'{punctuation_before}{processed_word}{punctuation_after}')
        else:
            # 普通单词
            if clean_word.lower() in PROTECTED_TERMS_LOWER:
                protected = PROTECTED_TERMS_LOWER[clean_word.lower()]
                if add_braces:
                    result.append(f'{punctuation_before}{{{protected}}}{punctuation_after}')
                else:
                    result.append(f'{punctuation_before}{protected}{punctuation_after}')
            elif is_likely_method_name(clean_word):
                if add_braces:
                    result.append(f'{punctuation_before}{{{clean_word}}}{punctuation_after}')
                else:
                    result.append(f'{punctuation_before}{clean_word}{punctuation_after}')
            else:
                if title_mode == 'title':
                    is_start = (i == 0)
                    is_after_colon = False
                    if i > 0 and ':' in words[i-1]:
                        is_after_colon = True
                    
                    if clean_word.lower() in LOWERCASE_WORDS and not is_start and not is_after_colon:
                        result.append(f'{punctuation_before}{clean_word.lower()}{punctuation_after}')
                    else:
                        capitalized = clean_word[0].upper() + clean_word[1:].lower() if len(clean_word) > 1 else clean_word.upper()
                        result.append(f'{punctuation_before}{capitalized}{punctuation_after}')
                elif title_mode == 'sentence':
                    is_start = (i == 0)
                    is_after_colon = False
                    if i > 0 and ':' in words[i-1]:
                        is_after_colon = True
                    
                    if is_start or is_after_colon:
                        capitalized = clean_word[0].upper() + clean_word[1:].lower() if len(clean_word) > 1 else clean_word.upper()
                        result.append(f'{punctuation_before}{capitalized}{punctuation_after}')
                    else:
                        result.append(f'{punctuation_before}{clean_word.lower()}{punctuation_after}')
                else:
                    result.append(word)
    
    final_result = ' '.join(result)
    final_result = final_result.replace(dash_placeholder, '-')
    
    for placeholder, original in braced_content.items():
        final_result = final_result.replace(placeholder, original)
    
    return final_result


def process_word_with_placeholders(word_part: str, word_idx: int, part_idx: int, 
                                   all_words: list, title_mode: str, 
                                   braced_content: dict) -> str:
    """
    处理包含 placeholder 的单个单词部分
    
    核心逻辑：
    - 单词中所有 placeholder 之间和之后的字母都保持小写
    - 只有单词开头（第一个 placeholder 之前）的文本才需要首字母大写
    """
    placeholder_matches = list(re.finditer(r'<<<BRACED_\d+>>>', word_part))
    
    if not placeholder_matches:
        # 没有 placeholder，正常处理
        if title_mode == 'title':
            is_start = (word_idx == 0 and part_idx == 0)
            is_after_colon = False
            if word_idx > 0 and ':' in all_words[word_idx-1]:
                is_after_colon = True
            
            if word_part.lower() in LOWERCASE_WORDS and not is_start and not is_after_colon:
                return word_part.lower()
            else:
                return word_part[0].upper() + word_part[1:].lower() if len(word_part) > 1 else word_part.upper()
        elif title_mode == 'sentence':
            is_start = (word_idx == 0 and part_idx == 0)
            is_after_colon = False
            if word_idx > 0 and ':' in all_words[word_idx-1]:
                is_after_colon = True
            
            if is_start or is_after_colon:
                return word_part[0].upper() + word_part[1:].lower() if len(word_part) > 1 else word_part.upper()
            else:
                return word_part.lower()
        else:
            return word_part
    
    # ✅ 有 placeholder 的情况
    parts = []
    last_end = 0
    
    for match_idx, match in enumerate(placeholder_matches):
        before_text = word_part[last_end:match.start()]
        
        if before_text:
            if match_idx == 0:
                # ✅ 第一个 placeholder 之前的文本：需要首字母大写
                if title_mode == 'title':
                    is_start = (word_idx == 0 and part_idx == 0)
                    is_after_colon = False
                    if word_idx > 0 and ':' in all_words[word_idx-1]:
                        is_after_colon = True
                    
                    if before_text.lower() in LOWERCASE_WORDS and not is_start and not is_after_colon:
                        parts.append(before_text.lower())
                    else:
                        parts.append(before_text[0].upper() + before_text[1:].lower() if len(before_text) > 1 else before_text.upper())
                elif title_mode == 'sentence':
                    is_start = (word_idx == 0 and part_idx == 0)
                    is_after_colon = False
                    if word_idx > 0 and ':' in all_words[word_idx-1]:
                        is_after_colon = True
                    
                    if is_start or is_after_colon:
                        parts.append(before_text[0].upper() + before_text[1:].lower() if len(before_text) > 1 else before_text.upper())
                    else:
                        parts.append(before_text.lower())
                else:
                    parts.append(before_text)
            else:
                # ✅ placeholder 之间的文本：全部小写（是单词的中间部分）
                parts.append(before_text.lower())
        
        parts.append(match.group(0))
        last_end = match.end()
    
    # ✅ 最后一个 placeholder 之后的文本：全部小写
    after_text = word_part[last_end:]
    if after_text:
        parts.append(after_text.lower())
    
    return ''.join(parts)

def capitalize_venue_name(text: str) -> str:
    """
    Capitalize venue names (journal/booktitle) using title case.
    """
    text_lower = text.lower()
    if text_lower in PRESERVE_TERMS_LOWER:
        return PRESERVE_TERMS_LOWER[text_lower]
    
    # Preserve specific terms
    preserve_placeholders = {}
    preserve_counter = 0
    text_for_processing = text
    sorted_preserve_terms = sorted(PRESERVE_TERMS, key=len, reverse=True)
    
    for preserve_term in sorted_preserve_terms:
        pattern = re.compile(re.escape(preserve_term), re.IGNORECASE)
        matches = list(pattern.finditer(text_for_processing))
        for match in reversed(matches):
            placeholder = f'<<<PRESERVE_{preserve_counter}>>>'
            preserve_placeholders[placeholder] = preserve_term
            text_for_processing = (
                text_for_processing[:match.start()] + 
                placeholder + 
                text_for_processing[match.end():]
            )
            preserve_counter += 1
    
    # Replace dashes with special placeholders
    dash_placeholder = '<<<DASH>>>'
    text_for_processing = re.sub(r'[-–—]', dash_placeholder, text_for_processing)
    
    words = text_for_processing.split()
    result = []
    
    for i, word in enumerate(words):
        # Restore preserved terms
        if word.startswith('<<<PRESERVE_') and word.endswith('>>>'):
            result.append(preserve_placeholders[word])
            continue
        
        clean_word = word.strip('.,;:!?()[]{}"\'-–—')
        punctuation_before = word[:len(word) - len(word.lstrip('.,;:!?()[]{}"\'-–—'))]
        punctuation_after = word[len(clean_word) + len(punctuation_before):]
        
        if '<<<PRESERVE_' in clean_word:
            for placeholder, original in preserve_placeholders.items():
                word = word.replace(placeholder, original)
            result.append(word)
            continue
        
        if not clean_word:
            result.append(word)
            continue
        
        # Check if word contains dash placeholder
        if dash_placeholder in clean_word:
            parts = clean_word.split(dash_placeholder)
            processed_parts = []
            
            for j, part in enumerate(parts):
                if not part:
                    continue
                
                if part.lower() in PROTECTED_TERMS_LOWER:
                    processed_parts.append(PROTECTED_TERMS_LOWER[part.lower()])
                elif is_likely_method_name(part):
                    processed_parts.append(part)
                else:
                    # Title case for venue names
                    is_start = (i == 0 and j == 0)
                    
                    if part.lower() in LOWERCASE_WORDS and not is_start:
                        processed_parts.append(part.lower())
                    else:
                        capitalized = part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()
                        processed_parts.append(capitalized)
            
            processed_word = '-'.join(processed_parts)
            result.append(f'{punctuation_before}{processed_word}{punctuation_after}')
        else:
            # Check for slash-separated parts (e.g., "A/B/C", "a/b/c")
            if '/' in clean_word:
                slash_parts = clean_word.split('/')
                processed_slash_parts = []
                
                for slash_part in slash_parts:
                    if not slash_part:
                        processed_slash_parts.append('')
                        continue
                    
                    if slash_part.lower() in PROTECTED_TERMS_LOWER:
                        processed_slash_parts.append(PROTECTED_TERMS_LOWER[slash_part.lower()])
                    elif is_likely_method_name(slash_part):
                        processed_slash_parts.append(slash_part)
                    else:
                        # For single uppercase letters, keep them uppercase
                        if len(slash_part) == 1 and slash_part.isupper():
                            processed_slash_parts.append(slash_part.upper())
                        else:
                            capitalized = slash_part[0].upper() + slash_part[1:].lower() if len(slash_part) > 1 else slash_part.upper()
                            processed_slash_parts.append(capitalized)
                
                processed_word = '/'.join(processed_slash_parts)
                result.append(f'{punctuation_before}{processed_word}{punctuation_after}')
            else:
                if clean_word.lower() in PROTECTED_TERMS_LOWER:
                    standardized = PROTECTED_TERMS_LOWER[clean_word.lower()]
                    result.append(f'{punctuation_before}{standardized}{punctuation_after}')
                    continue
                
                if is_likely_method_name(clean_word):
                    result.append(word)
                    continue
                
                # For single letters, always uppercase (e.g., "A" in "Journal of America A")
                if len(clean_word) == 1:
                    result.append(f'{punctuation_before}{clean_word.upper()}{punctuation_after}')
                    continue
                
                # Title case for venue names
                is_start = (i == 0)
                
                if clean_word.lower() in LOWERCASE_WORDS and not is_start:
                    result.append(f'{punctuation_before}{clean_word.lower()}{punctuation_after}')
                else:
                    capitalized = clean_word[0].upper() + clean_word[1:].lower() if len(clean_word) > 1 else clean_word.upper()
                    result.append(f'{punctuation_before}{capitalized}{punctuation_after}')
    
    final_result = ' '.join(result)
    
    # Restore dash placeholders
    final_result = final_result.replace(dash_placeholder, '-')
    
    return final_result

def smart_process_title(title: str, title_mode: str = 'title') -> str:
    """Process title field with colon handling and period removal"""
    if title_mode == 'keep':
        # Keep as-is mode
        return title
    
    # First, remove trailing period if present
    title, period_removed = remove_trailing_period(title)
    
    if ':' in title:
        colon_index = title.index(':')
        prefix = title[:colon_index].strip()
        suffix = title[colon_index + 1:].strip()
        
        prefix_processed = protect_proper_nouns(prefix, add_braces=True, title_mode=title_mode)
        suffix_processed = protect_proper_nouns(suffix, add_braces=True, title_mode=title_mode)
        return f'{prefix_processed}: {suffix_processed}'
    else:
        return protect_proper_nouns(title, add_braces=True, title_mode=title_mode)

def process_venue_field(text: str, venue_mode: str = 'format', venue_standardize_mode: str = 'none') -> Tuple[str, bool]:
    """
    Process journal/booktitle field.
    
    Args:
        text: Original venue text
        venue_mode: 'format' (apply title case) or 'keep' (keep as-is)
        venue_standardize_mode: 'none', 'full', 'abbr', or 'acronym'
    
    Returns:
        (processed_text, was_changed)
    """
    # Try to standardize to a canonical form if requested
    if venue_standardize_mode and venue_standardize_mode != 'none':
        standardized = standardize_venue_name(text, venue_standardize_mode, VENUE_LOOKUP, VENUE_BY_ID, VENUE_PATTERNS)
        if standardized:
            return standardized, standardized != text

    if venue_mode == 'keep':
        return text, False
    
    # Apply title case capitalization
    formatted = capitalize_venue_name(text)
    changed = (formatted != text)
    return formatted, changed

def extract_entry_key(entry: str) -> str:
    """Extract the citation key from a BibTeX entry (handles newlines)"""
    match = re.search(r'@\w+\s*\{\s*([^,\s]+)', entry, re.DOTALL)
    return match.group(1) if match else 'unknown'

def extract_title_from_entry(entry: str) -> str:
    """Extract title text from a BibTeX entry (best-effort)."""
    # Brace-delimited
    brace_match = re.search(
        r'\btitle\b\s*=\s*\{((?:[^{}]|\{[^}]*\})*)\}',
        entry,
        re.IGNORECASE | re.DOTALL,
    )
    if brace_match:
        return re.sub(r'\s+', ' ', brace_match.group(1)).strip()
    # Quote-delimited
    quote_match = re.search(
        r'\btitle\b\s*=\s*"([^"]*)"',
        entry,
        re.IGNORECASE | re.DOTALL,
    )
    if quote_match:
        return re.sub(r'\s+', ' ', quote_match.group(1)).strip()
    return ""

def extract_entry_type(entry: str) -> str:
    """Extract the entry type from a BibTeX entry"""
    match = re.search(r'@(\w+)\s*\{', entry, re.DOTALL)
    return match.group(1).lower() if match else 'unknown'

def iter_top_level_field_assignments(entry: str):
    """
    Yield top-level BibTeX field assignments as:
      (field_name, name_start, name_end, value_start, value_end)

    This is a best-effort parser that skips over quoted/braced values so we
    don't accidentally match patterns like "journal=" inside URLs or abstracts.
    """
    if not entry:
        return

    header_match = re.search(r'@\w+\s*\{\s*[^,]+,', entry, re.DOTALL)
    i = header_match.end() if header_match else 0
    n = len(entry)

    while i < n:
        while i < n and entry[i] in " \t\r\n,":
            i += 1
        if i >= n or entry[i] == '}':
            break

        name_start = i
        while i < n and (entry[i].isalnum() or entry[i] in "_-"):
            i += 1
        name_end = i
        if name_end <= name_start:
            i += 1
            continue
        field_name = entry[name_start:name_end]

        while i < n and entry[i] in " \t\r\n":
            i += 1
        if i >= n or entry[i] != '=':
            continue
        i += 1

        while i < n and entry[i] in " \t\r\n":
            i += 1
        value_start = i

        brace_depth = 0
        in_quotes = False
        while i < n:
            ch = entry[i]
            if in_quotes:
                if ch == '"' and (i == 0 or entry[i - 1] != '\\'):
                    in_quotes = False
                i += 1
                continue
            if ch == '"':
                in_quotes = True
                i += 1
                continue
            if ch == '{':
                brace_depth += 1
                i += 1
                continue
            if ch == '}':
                if brace_depth > 0:
                    brace_depth -= 1
                    i += 1
                    continue
                break
            if ch == ',' and brace_depth == 0:
                break
            i += 1

        value_end = i
        yield field_name, name_start, name_end, value_start, value_end

def has_field_assignment(entry: str, field_name: str) -> bool:
    """Return True if the entry contains an assignment for the given field name."""
    if not entry or not field_name:
        return False
    target = field_name.lower()
    for name, _, _, _, _ in iter_top_level_field_assignments(entry):
        if name.lower() == target:
            return True
    return False

def extract_field_value(entry: str, field_name: str) -> str:
    """Best-effort extraction of a field's value (brace or quote delimited)."""
    if not entry or not field_name:
        return ''
    target = field_name.lower()
    for name, _, _, value_start, value_end in iter_top_level_field_assignments(entry):
        if name.lower() != target:
            continue
        raw = entry[value_start:value_end].strip()
        if len(raw) >= 2 and ((raw[0] == '{' and raw[-1] == '}') or (raw[0] == '"' and raw[-1] == '"')):
            raw = raw[1:-1]
        return re.sub(r'\s+', ' ', raw).strip()
    return ''

def rename_field_assignment(entry: str, src_field: str, dst_field: str) -> str:
    """Rename a field assignment key (e.g., journal -> booktitle) without changing its value."""
    if not entry or not src_field or not dst_field:
        return entry
    target = src_field.lower()
    for name, name_start, name_end, _, _ in iter_top_level_field_assignments(entry):
        if name.lower() == target:
            return entry[:name_start] + dst_field + entry[name_end:]
    return entry

def fix_venue_field_names(entry: str, log: ProcessingLog, entry_key: str, entry_type: str) -> Tuple[str, bool]:
    """
    Fix common BibTeX field misuse:
    - @inproceedings: journal -> booktitle (when booktitle missing)
    - @article: booktitle -> journal (when journal missing)
    """
    etype = (entry_type or '').lower()
    if etype in {'inproceedings', 'conference'}:
        if has_field_assignment(entry, 'journal') and not has_field_assignment(entry, 'booktitle'):
            value = extract_field_value(entry, 'journal')
            updated = rename_field_assignment(entry, 'journal', 'booktitle')
            if updated != entry:
                log.add_change(entry_key, 'field_renamed', 'journal -> booktitle', value, value)
                return updated, True
    if etype == 'article':
        if has_field_assignment(entry, 'booktitle') and not has_field_assignment(entry, 'journal'):
            value = extract_field_value(entry, 'booktitle')
            updated = rename_field_assignment(entry, 'booktitle', 'journal')
            if updated != entry:
                log.add_change(entry_key, 'field_renamed', 'booktitle -> journal', value, value)
                return updated, True
    return entry, False

def remove_unwanted_fields(entry: str, keep_fields: Set[str], log: ProcessingLog, entry_key: str, entry_type: str) -> Tuple[str, bool]:
    """Remove fields not in KEEP_FIELDS (only for specified entry types)"""
    should_remove = entry_type.lower() in REMOVE_FIELDS_FOR_TYPES
    
    if not should_remove:
        return entry, False
    
    lines = entry.split('\n')
    result_lines = [lines[0]]
    modified = False
    
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or stripped == '}':
            result_lines.append(line)
            continue
        
        field_match = re.match(r'(\w+)\s*=', stripped)
        if field_match:
            field_name = field_match.group(1).lower()
            if field_name in keep_fields:
                result_lines.append(line)
            else:
                log.add_change(entry_key, 'removed_field', field_name, stripped, "")
                modified = True
        else:
            result_lines.append(line)
    
    return '\n'.join(result_lines), modified

def format_fields(entry: str, log: ProcessingLog, entry_key: str, title_mode: str = 'title', venue_mode: str = 'format', venue_standardize_mode: str = 'none') -> Tuple[str, bool]:
    """
    Format title, journal, and booktitle fields.
    
    NOTE: This function processes title/journal/booktitle fields for ALL entry types.
    
    IMPORTANT: Handles both {...} and "..." delimiters, and multi-line field values.
    """
    modified = False
    result = entry
    
    # Process each field type (title, journal, booktitle)
    for field_type in ['title', 'journal', 'booktitle']:
        # Pattern 1: Match fields with curly braces {...}
        pattern_braces = rf'(?<!\w)(\s*)({field_type})(?!\w)(\s*=\s*)(\{{)((?:[^{{}}]|\{{[^}}]*\}})*?)(\}})'
        
        # Pattern 2: Match fields with double quotes "..."
        pattern_quotes = rf'(?<!\w)(\s*)({field_type})(?!\w)(\s*=\s*)(")([^"]*?)(")'
        
        colon_detected = False
        
        def mark_colon(clean_text, formatted_text):
            nonlocal colon_detected
            if not colon_detected and (':' in clean_text or ':' in formatted_text):
                log.colon_titles_count += 1
                colon_detected = True
        
        def replace_field_braces(match):
            nonlocal modified
            indent = match.group(1)
            field_name = match.group(2)
            equals_part = match.group(3)
            opening = match.group(4)  # {
            field_value = match.group(5)
            closing = match.group(6)  # }
            
            field_name_lower = field_name.lower()
            
            # Normalize whitespace
            clean_value = re.sub(r'\s+', ' ', field_value.strip())
            
            if field_name_lower == 'title':
                formatted_value = smart_process_title(clean_value, title_mode)
                
                # Count period removals only when output actually drops the trailing period
                if clean_value.endswith('.') and not formatted_value.endswith('.'):
                    log.period_removed_count += 1
                
                # Track titles containing colons even if unchanged
                mark_colon(clean_value, formatted_value)
            elif field_name_lower in ['journal', 'booktitle']:
                formatted_value, _ = process_venue_field(clean_value, venue_mode, venue_standardize_mode)
            else:
                formatted_value = clean_value
            
            if clean_value != formatted_value:
                log.add_change(entry_key, 'field_modified', field_name_lower, clean_value, formatted_value)
                modified = True
            
            return f'{indent}{field_name}{equals_part}{opening}{formatted_value}{closing}'
        
        def replace_field_quotes(match):
            nonlocal modified
            indent = match.group(1)
            field_name = match.group(2)
            equals_part = match.group(3)
            opening = match.group(4)  # "
            field_value = match.group(5)
            closing = match.group(6)  # "
            
            field_name_lower = field_name.lower()
            
            # Normalize whitespace
            clean_value = re.sub(r'\s+', ' ', field_value.strip())
            
            if field_name_lower == 'title':
                formatted_value = smart_process_title(clean_value, title_mode)
                
                # Count period removals only when output actually drops the trailing period
                if clean_value.endswith('.') and not formatted_value.endswith('.'):
                    log.period_removed_count += 1
                
                # Track titles containing colons even if unchanged
                mark_colon(clean_value, formatted_value)
            elif field_name_lower in ['journal', 'booktitle']:
                formatted_value, _ = process_venue_field(clean_value, venue_mode, venue_standardize_mode)
            else:
                formatted_value = clean_value
            
            if clean_value != formatted_value:
                log.add_change(entry_key, 'field_modified', field_name_lower, clean_value, formatted_value)
                modified = True
            
            # Convert to braces for consistency
            return f'{indent}{field_name}{equals_part}{{{formatted_value}}}'
        
        # Apply both patterns
        result = re.sub(pattern_braces, replace_field_braces, result, flags=re.IGNORECASE | re.DOTALL)
        result = re.sub(pattern_quotes, replace_field_quotes, result, flags=re.IGNORECASE | re.DOTALL)
    
    return result, modified

def detect_duplicate_groups_in_content(content: str) -> List[Dict[str, List[Dict[str, str]]]]:
    """Detect duplicate entries by normalized title or by key."""
    parsed = []
    key_occurrence: Dict[str, int] = {}
    entry_order = 0
    entries = re.split(r'(?=@\w+\s*\{)', content)
    for entry in entries:
        entry_strip = entry.strip()
        if not entry_strip or not entry_strip.startswith('@'):
            continue
        key = extract_entry_key(entry_strip)
        try:
            occ = key_occurrence.get(key, 0)
        except Exception:
            occ = 0
        key_occurrence[key] = occ + 1
        title = extract_title_from_entry(entry_strip)
        norm_title = normalize_title_for_duplicates(title) if title else ''
        if not key and not norm_title:
            continue
        parsed.append({
            'key': key,
            'title': title,
            'normalized_title': norm_title,
            'entry': entry_strip,
            'occurrence': occ,
            'order': entry_order
        })
        entry_order += 1
    return build_duplicate_groups(parsed)


def merge_entries_in_content(content: str, merges: List[Dict[str, str]], line_counts: Dict[str, List[int]] = None, preserve_spacing: bool = True) -> str:
    """
    Merge entries in the content based on merges instructions.
    Each merge dict: {'master': key, 'merge_keys': [...], 'master_idx': optional occurrence index}.
    
    Default behavior (used by auto-merge):
    - Treat the first entry in each duplicate group as the master.
    - Merge all other duplicates into that master.
    - Keep the master's existing values when there is a conflict; only fill
      missing/empty fields from the duplicates (e.g., master's year wins if set).
    """
    entries = re.split(r'(?=@\w+\s*\{)', content)
    parsed = []
    key_occurrence = {}
    entry_order = 0
    for raw in entries:
        entry_strip = raw.strip()
        if not entry_strip:
            continue
        if not entry_strip.startswith('@'):
            parsed.append({'key': None, 'raw': raw})
            continue
        key = extract_entry_key(entry_strip)
        occ = key_occurrence.get(key, 0)
        key_occurrence[key] = occ + 1
        original_line_count = None
        if line_counts and key in line_counts and occ < len(line_counts[key]):
            original_line_count = line_counts[key][occ]
        if original_line_count is None:
            original_line_count = count_lines(raw)
        parsed.append({
            'key': key,
            'raw': raw,
            'line_count': original_line_count,
            'newline_count': raw.count('\n'),
            'occurrence': occ,
            'order_idx': entry_order
        })
        entry_order += 1
    
    # Helper to parse fields (simple)
    def parse_fields(entry_text: str) -> Tuple[str, Dict[str, str], str]:
        entry_type = extract_entry_type(entry_text)
        fields = {}
        body_match = re.search(r'@\w+\s*\{\s*[^,]+,(.*)\}\s*$', entry_text, re.DOTALL)
        body = body_match.group(1) if body_match else ''
        pattern = re.compile(r'(\w+)\s*=\s*(\{(?:[^{}]|\{[^}]*\})*\}|"[^"]*")\s*,?', re.DOTALL)
        for m in pattern.finditer(body):
            name = m.group(1).lower()
            value = m.group(2).strip()
            # strip outer braces/quotes
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            fields[name] = value
        return entry_type, fields, body
    
    def has_value(val: str) -> bool:
        """Return True if the field value is non-empty after stripping whitespace."""
        if val is None:
            return False
        if isinstance(val, str):
            return bool(val.strip())
        return True
    
    def serialize_entry(entry_type: str, key: str, fields: Dict[str, str]) -> str:
        lines = [f"@{entry_type}{{{key},"]
        for name, value in fields.items():
            lines.append(f"  {name} = {{{value}}},")
        if fields and lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]
        lines.append("}")
        return '\n'.join(lines)

    def sanitize_field_value(value: str) -> str:
        if value is None:
            return ''
        # Ensure field values don't introduce extra newlines when we need to lock line budgets.
        return ' '.join(str(value).replace('\r', '').split())

    def serialize_entry_one_line(entry_type: str, key: str, fields: Dict[str, str], sanitize_values: bool = False) -> str:
        parts = []
        for name, value in fields.items():
            v = sanitize_field_value(value) if sanitize_values else value
            parts.append(f"{name} = {{{v}}}")
        body = ', '.join(parts)
        if body:
            return f"@{entry_type}{{{key}, {body}}}"
        return f"@{entry_type}{{{key},}}"

    def serialize_entry_capped(entry_type: str, key: str, fields: Dict[str, str], max_lines: int) -> str:
        if max_lines is None or max_lines <= 0:
            return serialize_entry(entry_type, key, fields)

        pretty = serialize_entry(entry_type, key, fields)
        if count_lines(pretty) <= max_lines:
            return pretty

        # Need to compress: collapse internal whitespace/newlines in values first.
        field_parts = []
        for name, value in fields.items():
            field_parts.append(f"{name} = {{{sanitize_field_value(value)}}}")

        if max_lines <= 1:
            return serialize_entry_one_line(entry_type, key, fields, sanitize_values=True)

        if max_lines == 2:
            header = f"@{entry_type}{{{key},"
            if not field_parts:
                return f"{header}\n}}"
            body = ', '.join(field_parts)
            return f"{header}\n  {body}}}"

        # max_lines >= 3
        header = f"@{entry_type}{{{key},"
        footer = "}"
        if not field_parts:
            return f"{header}\n{footer}"

        body_lines = max_lines - 2
        body_lines = min(body_lines, len(field_parts))
        base, rem = divmod(len(field_parts), body_lines)
        chunks = []
        cursor = 0
        for i in range(body_lines):
            size = base + (1 if i < rem else 0)
            chunks.append(field_parts[cursor:cursor + size])
            cursor += size

        lines = [header]
        for idx, chunk in enumerate(chunks):
            line = ', '.join(chunk)
            if idx < len(chunks) - 1:
                line += ','
            lines.append(f"  {line}")
        lines.append(footer)
        return '\n'.join(lines)
    
    parsed_map = {}
    for item in parsed:
        if not item['key']:
            continue
        parsed_map.setdefault(item['key'], []).append(item)
    
    for merge in merges:
        master_key = merge['master']
        merge_items = merge.get('merge_items') or []
        allowed_occurrences: Dict[str, Set[int]] = {}
        for item in merge_items:
            key = item.get('key')
            if not key:
                continue
            try:
                occ_val = int(item.get('occurrence', 0))
            except Exception:
                continue
            allowed_occurrences.setdefault(key, set()).add(occ_val)
        merge_keys = merge.get('merge_keys') or list(allowed_occurrences.keys())
        master_idx = merge.get('master_idx', 0)
        if master_key not in parsed_map or not parsed_map[master_key]:
            continue
        try:
            master_idx_int = int(master_idx)
        except Exception:
            master_idx_int = 0
        if master_idx_int < 0 or master_idx_int >= len(parsed_map[master_key]):
            master_idx_int = 0
        master_item = parsed_map[master_key][master_idx_int]
        master_order = merge.get('master_order')
        if master_order is not None:
            for idx, candidate in enumerate(parsed_map[master_key]):
                if candidate.get('order_idx') == master_order:
                    master_item = candidate
                    master_idx_int = idx
                    break
        if master_item.get('remove'):
            # Master entry was already merged away somehow; fallback to first available
            master_idx_int = 0
            master_item = parsed_map[master_key][0]
        m_entry = master_item['raw']
        m_type, m_fields, _ = parse_fields(m_entry)
        
        for k in merge_keys:
            if k not in parsed_map or not parsed_map[k]:
                continue
            # Take the next available entry for this key (skip master item)
            candidate_list = parsed_map[k]
            for idx, other_item in enumerate(candidate_list):
                other_item = candidate_list[idx]
                if other_item.get('remove'):
                    continue
                if k == master_key and idx == master_idx_int:
                    # Skip the chosen master occurrence
                    continue
                allowed_set = allowed_occurrences.get(k)
                if allowed_set is not None and other_item.get('occurrence') not in allowed_set:
                    # Not part of this duplicate group; leave it untouched
                    continue
                other_entry = other_item['raw']
                _, o_fields, _ = parse_fields(other_entry)
                
                # Merge fields: keep master's existing values; fill missing/empty from others
                for fname, fval in o_fields.items():
                    master_val = m_fields.get(fname)
                    if not has_value(master_val) and has_value(fval):
                        m_fields[fname] = fval
                
                # Mark merged entry for placeholder removal; optionally preserve original spacing
                if preserve_spacing:
                    newline_count = other_item.get('newline_count')
                    if newline_count is None:
                        newline_count = other_item['raw'].count('\n')
                    placeholder_lines = max(1, newline_count)
                else:
                    placeholder_lines = 0  # Drop merged entries entirely when spacing preservation is off
                other_item['placeholder'] = '\n' * placeholder_lines
                other_item['remove'] = True

        # Update master entry text and pad to preserve original line count
        master_target_lines = master_item.get('line_count') or count_lines(master_item['raw'])
        raw_master = master_item.get('raw') or ''
        trailing_newlines = len(raw_master) - len(raw_master.rstrip('\n'))
        reserve_newlines = trailing_newlines if trailing_newlines > 0 else (1 if master_target_lines > 1 else 0)
        max_master_lines = max(1, master_target_lines - reserve_newlines)
        master_serialized = serialize_entry_capped(m_type, master_key, m_fields, max_master_lines)
        master_item['raw'] = pad_to_line_count(master_serialized, master_target_lines)
    
    # Rebuild content
    rebuilt = []
    for item in parsed:
        if item.get('key') and item.get('remove'):
            if item.get('placeholder'):
                rebuilt.append(item['placeholder'])
            continue
        rebuilt.append(item['raw'])
    
    return ''.join(rebuilt)


def process_bibtex_content(content: str, keep_fields: Set[str], title_mode: str = 'title', venue_mode: str = 'format', venue_standardize_mode: str = 'none', auto_merge_duplicates: bool = False, merge_plan: List[Dict[str, Any]] = None) -> Tuple[str, ProcessingLog]:
    """Process entire BibTeX content"""
    log = ProcessingLog()
    log.keep_fields = keep_fields.copy()
    log.title_mode = title_mode
    log.venue_mode = venue_mode
    log.venue_standardize_mode = venue_standardize_mode
    
    entries = re.split(r'(?=@\w+\s*\{)', content)
    processed_entries = []
    duplicate_candidates: List[Dict[str, Any]] = []
    line_counts_by_key: Dict[str, List[int]] = {}
    key_occurrence: Dict[str, int] = {}
    entry_order = 0
    
    for entry in entries:
        original_entry = entry
        entry = entry.strip()
        if not entry:
            continue
            
        if entry.startswith('@'):
            entry_type = extract_entry_type(entry)
            entry_key = extract_entry_key(entry)
            occ = key_occurrence.get(entry_key, 0)
            key_occurrence[entry_key] = occ + 1
            entry_line_count = count_lines(original_entry)
            line_counts_by_key.setdefault(entry_key, []).append(entry_line_count)
            log.add_entry(entry_type)
            entry_modified = False
            order_idx = entry_order
            entry_order += 1

            # Step 0: Fix common field misuse (e.g., inproceedings journal -> booktitle)
            entry, fields_fixed = fix_venue_field_names(entry, log, entry_key, entry_type)
            entry_modified = entry_modified or fields_fixed

            # Step 1: Remove unwanted fields (only for article/inproceedings)
            cleaned_entry, fields_removed = remove_unwanted_fields(entry, keep_fields, log, entry_key, entry_type)
            entry_modified = entry_modified or fields_removed
            
            # Step 2: Format title/journal/booktitle fields (for ALL entry types)
            formatted_entry, fields_formatted = format_fields(
                cleaned_entry,
                log,
                entry_key,
                title_mode,
                venue_mode,
                venue_standardize_mode,
            )
            entry_modified = entry_modified or fields_formatted
            
            # Duplicate detection: extract processed title and index
            extracted_title = extract_title_from_entry(formatted_entry)
            norm_title = normalize_title_for_duplicates(extracted_title) if extracted_title else ''
            if entry_key or norm_title:
                duplicate_candidates.append({
                    'key': entry_key,
                    'title': extracted_title,
                    'normalized_title': norm_title,
                    'entry': formatted_entry,
                    'occurrence': occ,
                    'order': order_idx
                })
            
            if entry_modified:
                log.mark_modified()
            
            # Pad entry to preserve original line count for alignment
            padded_entry = pad_to_line_count(formatted_entry, entry_line_count)
            processed_entries.append(padded_entry)
        else:
            processed_entries.append(original_entry)
    
    # Record duplicates in log (by title OR by key)
    log.duplicates = build_duplicate_groups(duplicate_candidates)
    
    processed_output = ''.join(processed_entries)
    log.pre_merge_output = processed_output
    log.duplicates_pre_merge = list(log.duplicates)
    
    # Build merge instructions:
    # - If a user-supplied plan exists, use it and skip the default auto-merge.
    # - Otherwise, fall back to the automatic \"first entry wins\" strategy.
    merges: List[Dict[str, Any]] = []
    if merge_plan is not None:
        for item in merge_plan:
            try:
                should_merge = bool(item.get('merge', True))
                if not should_merge:
                    continue
                master = item.get('master')
                merge_keys = item.get('merge_keys', [])
                merge_items_clean = []
                for mi in item.get('merge_items') or []:
                    key = mi.get('key')
                    if not key:
                        continue
                    try:
                        occ_val = int(mi.get('occurrence', 0))
                    except Exception:
                        occ_val = 0
                    m_order = mi.get('order')
                    try:
                        m_order = int(m_order) if m_order is not None else None
                    except Exception:
                        m_order = None
                    merge_items_clean.append({'key': key, 'occurrence': occ_val, 'order': m_order})
                if not master or (not merge_keys and not merge_items_clean):
                    continue
                merges.append({
                    'master': master,
                    'merge_keys': merge_keys,
                    'group_norm': item.get('group_norm', ''),
                    'master_idx': item.get('master_idx', 0),
                    'master_order': item.get('master_order'),
                    'merge_items': merge_items_clean
                })
            except Exception:
                # Defensive: skip malformed merge instructions
                continue
    elif auto_merge_duplicates and log.duplicates:
        for group in log.duplicates:
            entries_group = sorted(group['entries'], key=lambda e: e.get('order', 0))
            if len(entries_group) <= 1:
                continue
            master = entries_group[0].get('key')
            if not master:
                continue
            merge_keys = []
            seen_keys = set()
            merge_items = []
            for e in entries_group[1:]:
                key_val = e.get('key')
                if not key_val:
                    continue
                if key_val not in seen_keys:
                    merge_keys.append(key_val)
                    seen_keys.add(key_val)
                merge_items.append({
                    'key': key_val,
                    'occurrence': e.get('occurrence', 0),
                    'order': e.get('order')
                })
            merges.append({
                'master': master,
                'merge_keys': merge_keys,
                'group_norm': group.get('group_norm') or group.get('normalized_title', ''),
                'master_idx': entries_group[0].get('occurrence', 0),
                'master_order': entries_group[0].get('order'),
                'merge_items': merge_items
            })
    
    if merges:
        processed_output = merge_entries_in_content(
            processed_output,
            merges,
            line_counts=line_counts_by_key,
            preserve_spacing=True  # Keep original spacing so masters stay at their original positions
        )
        for merge in merges:
            log.add_change(merge['master'], 'merged_entry', 'title', merge.get('group_norm', ''), merge['merge_keys'])
        log.duplicates = detect_duplicate_groups_in_content(processed_output)
        log.duplicates_post_merge = list(log.duplicates)
    
    # Keep total line count aligned with the original input to avoid cursor/overlay drift in the UI
    processed_output = normalize_total_line_count(processed_output, count_lines(content))
    return processed_output, log

def process_bibtex_file(input_file: str, output_file: str, log_file: str = None, 
                       keep_fields: Set[str] = None, title_mode: str = 'title',
                       venue_mode: str = 'format', auto_merge_duplicates: bool = False) -> Tuple[bool, str, ProcessingLog]:
    """Process BibTeX file and generate output + log"""
    if keep_fields is None:
        keep_fields = DEFAULT_KEEP_FIELDS
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        processed_content, log = process_bibtex_content(
            content, keep_fields, title_mode, venue_mode, auto_merge_duplicates=auto_merge_duplicates
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        if log_file is None:
            log_file = output_file.rsplit('.', 1)[0] + '_processing_log.txt'
        
        log_content = log.generate_report()
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        summary = (
            f"Successfully processed:\n"
            f"  Input: {input_file}\n"
            f"  Output: {output_file}\n"
            f"  Log: {log_file}\n\n"
            f"Statistics:\n"
            f"  Total entries: {log.total_entries}\n"
            f"  Modified entries: {log.modified_entries}\n"
            f"  Unchanged entries: {log.total_entries - log.modified_entries}\n"
            f"  Title changes: {log.title_changes_count}\n"
            f"  Title periods removed: {log.period_removed_count}\n"
            f"  Journal changes: {log.journal_changes_count}\n"
            f"  Booktitle changes: {log.booktitle_changes_count}\n"
            f"  Fields removed: {log.removed_fields_count}"
        )
        
        merged_groups = [c for c in log.changes if c['type'] == 'merged_entry']
        if auto_merge_duplicates and merged_groups:
            summary += (
                f"\n  Duplicate merges applied: {len(merged_groups)} group(s) "
                f"(auto-merge duplicates; disable with --no-duplicate-merge)"
            )
        
        if log.colon_titles_count > 0:
            summary += (
                f"\n\n"
                f"{'!' * 70}\n"
                f"WARNING: {log.colon_titles_count} title(s) contain colons.\n"
                f"   Please manually verify custom names/methods before the colon\n"
                f"   (e.g., '{{NeRF}}', '{{HiDDeN}}', etc.) are correctly formatted!\n"
                f"{'!' * 70}"
            )
        
        if log.duplicates:
            dup_groups = len(log.duplicates)
            dup_entries = sum(len(item['entries']) for item in log.duplicates)
            summary += (
                f"\n\n"
                f"{'!' * 70}\n"
                f"POTENTIAL DUPLICATES DETECTED: {dup_groups} group(s), {dup_entries} entries.\n"
                f"   Use the GUI 'Resolve Duplicates' button to review/merge, "
                f"or CLI auto-merge (enabled by default).\n"
                f"{'!' * 70}"
            )
        
        return True, summary, log
    
    except FileNotFoundError:
        return False, f"Error: File not found: {input_file}", None
    except Exception as e:
        return False, f"Error processing file: {str(e)}", None
