#!/usr/bin/env python3
"""
Create an Overleaf-friendly minimal bundle for the paper.

Actions:
1) Scan all .tex files under <paper_dir> for \includegraphics paths
2) Copy all referenced images into <paper_dir>/figures/ with de-duplicated names
3) Rewrite the .tex files to use figures/<copied_name> paths
4) Create a ZIP archive containing only the necessary files for Overleaf:
   - main.tex
   - sections/*.tex
   - generated/*.tex
   - references.bib
   - figures/* (copied images)

Usage:
  python -m src.tools.make_overleaf_bundle \
    --paper-dir MobileDeepfakeDetection/paper \
    --zip-out MobileDeepfakeDetection/paper_overleaf_bundle.zip
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile


INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def find_tex_files(paper_dir: Path) -> List[Path]:
    files: List[Path] = []
    # main
    files.append(paper_dir / 'main.tex')
    # sections
    files.extend(sorted((paper_dir / 'sections').glob('*.tex')))
    # generated
    files.extend(sorted((paper_dir / 'generated').glob('*.tex')))
    # Only keep existing
    return [p for p in files if p.exists()]


def extract_graphics_paths(tex_content: str) -> List[str]:
    return [m.group(1).strip() for m in INCLUDE_RE.finditer(tex_content)]


def safe_name_from_path(orig: str) -> str:
    # Normalize and create a flat filename reflecting the original path
    # Replace separators and disallowed chars with '_'
    s = orig
    # Drop leading ../ or ./ sequences for cleanliness
    while s.startswith('../') or s.startswith('./'):
        s = s[3:] if s.startswith('../') else s[2:]
    s = s.strip()
    # Replace slashes and backslashes
    s = s.replace('/', '_').replace('\\', '_')
    # Keep alnum, underscore, dot, dash; replace others
    s = re.sub(r'[^A-Za-z0-9_.\-]', '_', s)
    return s


def resolve_source_path(paper_dir: Path, rel_path: str) -> Path:
    # Treat rel_path relative to paper_dir
    p = (paper_dir / rel_path).resolve()
    if p.exists():
        return p
    # Try relative to repo root (paper_dir.parent)
    alt = (paper_dir.parent / rel_path).resolve()
    if alt.exists():
        return alt
    return p  # may not exist


def copy_and_rewrite(paper_dir: Path) -> Tuple[Dict[str, str], List[Path]]:
    figures_dir = paper_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    path_map: Dict[str, str] = {}
    touched: List[Path] = []

    for tex_file in find_tex_files(paper_dir):
        content = tex_file.read_text(encoding='utf-8')
        matches = extract_graphics_paths(content)
        changed = content

        for rel in matches:
            if rel in path_map:
                new_rel = path_map[rel]
            else:
                src = resolve_source_path(paper_dir, rel)
                new_name = safe_name_from_path(rel)
                dst = figures_dir / new_name
                if src.exists():
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                    new_rel = f'figures/{new_name}'
                else:
                    # If source missing, keep original path; Overleaf will error, but we don't break the file
                    new_rel = rel
                path_map[rel] = new_rel

            # Replace only exact matches inside includegraphics braces
            # Use regex sub with function to avoid replacing outside
            def _repl(m: re.Match) -> str:
                inner = m.group(1).strip()
                return m.group(0).replace(inner, path_map.get(inner, inner))

            changed = INCLUDE_RE.sub(_repl, changed)

        if changed != content:
            tex_file.write_text(changed, encoding='utf-8')
            touched.append(tex_file)

    return path_map, touched


def make_zip(paper_dir: Path, zip_out: Path) -> None:
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zf:
        # required files
        keep_list: List[Path] = []
        keep_list.append(paper_dir / 'main.tex')
        keep_list.append(paper_dir / 'references.bib')
        keep_list.extend(sorted((paper_dir / 'sections').glob('*.tex')))
        keep_list.extend(sorted((paper_dir / 'generated').glob('*.tex')))
        keep_list.extend(sorted((paper_dir / 'figures').glob('*')))
        # Filter existing
        keep_list = [p for p in keep_list if p.exists()]
        for p in keep_list:
            arcname = p.relative_to(paper_dir).as_posix()
            zf.write(p, arcname)


def main() -> int:
    ap = argparse.ArgumentParser(description='Make Overleaf-friendly paper bundle')
    ap.add_argument('--paper-dir', default='MobileDeepfakeDetection/paper')
    ap.add_argument('--zip-out', default='MobileDeepfakeDetection/paper_overleaf_bundle.zip')
    args = ap.parse_args()

    paper_dir = Path(args.paper_dir).resolve()
    zip_out = Path(args.zip_out).resolve()

    if not paper_dir.exists():
        raise SystemExit(f"Paper dir not found: {paper_dir}")

    path_map, touched = copy_and_rewrite(paper_dir)
    print(f"Rewrote {len(touched)} .tex files; copied {len(path_map)} figure references")
    make_zip(paper_dir, zip_out)
    print(f"Wrote Overleaf bundle: {zip_out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

