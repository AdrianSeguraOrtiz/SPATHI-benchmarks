#!/usr/bin/env python3
"""Check vector outputs and optionally reproduce them in a source-only folder.

    python verify_build.py
    python verify_build.py --rebuild

The clean rebuild intentionally receives only build_figure.py and the two JSON
inputs. Neither the reference PNG nor any pre-rendered SVG/PDF/PNG is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    cfg = json.loads((ROOT / 'figure_config.json').read_text(encoding='utf-8'))
    stem = cfg['output_basename']
    report = {'outputs': {}, 'source_only_rebuild': None}
    path = ROOT / (stem + '.svg')
    tree = ET.parse(path)
    images = []
    external = []
    for node in tree.getroot().iter():
        tag = node.tag.rsplit('}', 1)[-1]
        if tag in {'image', 'foreignObject', 'script'}:
            images.append(tag)
        for key, value in node.attrib.items():
            if key.rsplit('}', 1)[-1] == 'href' and not value.startswith('#'):
                external.append(value)
    if images or external:
        raise RuntimeError(f'{path.name}: raster/foreign/external content detected')
    report['outputs'][path.name] = {'sha256': digest(path), 'raster_images': 0,
                                   'external_resource_links': 0,
                                   'font_dependency': cfg['font_family']}
    path = ROOT / (stem + '.pdf')
    with fitz.open(path) as pdf:
        if len(pdf) != 1:
            raise RuntimeError('Expected a one-page figure')
        page = pdf[0]
        if page.get_images(full=True):
            raise RuntimeError('The PDF contains raster images')
        if 'SPATHI-benchmarks (FULL)' not in page.get_text():
            raise RuntimeError('The PDF does not contain the expected selectable title')
        report['outputs'][path.name] = {'sha256': digest(path), 'raster_images': 0,
                                       'selectable_text': True}
    checks = json.loads((ROOT / 'layout_checks.json').read_text())
    if checks['text_outside_regions'] or checks['text_box_overlaps']:
        raise RuntimeError('Layout check failed')
    report['layout_checks'] = checks
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--font-dir', type=Path)
    args = parser.parse_args()
    try:
        report = verify()
        if args.rebuild:
            cfg = json.loads((ROOT / 'figure_config.json').read_text())
            stem = cfg['output_basename']
            with tempfile.TemporaryDirectory(prefix='spathi-figure-clean-') as tmp:
                clean = Path(tmp)
                for filename in ['build_figure.py', 'figure_config.json', 'figure_text.json']:
                    shutil.copyfile(ROOT / filename, clean / filename)
                command = [sys.executable, str(clean / 'build_figure.py'),
                           '--out-dir', str(clean / 'out')]
                if args.font_dir:
                    command += ['--font-dir', str(args.font_dir.resolve())]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode:
                    raise RuntimeError(result.stdout + '\n' + result.stderr)
                names = [stem + suffix for suffix in
                         ['.svg', '.pdf', '.png']]
                comparison = {name: digest(ROOT/name) == digest(clean/'out'/name) for name in names}
                report['source_only_rebuild'] = {
                    'copied_inputs': ['build_figure.py', 'figure_config.json', 'figure_text.json'],
                    'reference_png_present': False,
                    'byte_identical_to_delivered_outputs': comparison,
                }
                if not all(comparison.values()):
                    raise RuntimeError('A regenerated output differs. Compare the font hashes and '
                                       'versions in build_environment.json. This check assumes the '
                                       'default 300 dpi and the delivered configuration.')
        path = ROOT / 'verification.json'
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Verification failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
