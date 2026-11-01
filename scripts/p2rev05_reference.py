#!/usr/bin/env python3
"""PROMPT P2REV-05 Job 4a - build PolarityPaper/revision/reference.docx.

Body: Times New Roman 11 pt. Table style ("Table", the style pandoc applies):
9 pt text, thin top and bottom rules and a rule under the header row, no
vertical lines, header-row repeat, 2 pt cell padding. Caption styles
("Table Caption", "Image Caption", "Caption"): 9 pt italic; pandoc places
table captions above the table.
"""
from pathlib import Path

import pypandoc
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'PolarityPaper' / 'revision' / 'reference.docx'

PROBE = """Text paragraph.

| a | b |
|---|---|
| 1 | 2 |

: A table caption

![An image caption](PROBE_IMG)
"""


def el(tag, **attrs):
    from docx.oxml import OxmlElement
    e = OxmlElement(tag if ':' in tag else f'w:{tag}')
    for k, v in attrs.items():
        e.set(qn(f'w:{k}'), str(v))
    return e


def set_font(style, name='Times New Roman', size=11, italic=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.italic = italic or None
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = el('rFonts')
        rpr.append(rfonts)
    for a in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        rfonts.set(qn(f'w:{a}'), name)


def main():
    img = ROOT / 'PolarityPaper' / 'revision' / 'figures' / \
        'schematic_tangent_sign.png'
    probe_md = PROBE.replace('PROBE_IMG', img.as_posix())
    pypandoc.convert_text(probe_md, 'docx', format='md',
                          outputfile=str(OUT))
    doc = Document(str(OUT))
    styles = doc.styles

    set_font(styles['Normal'], size=11)
    for name in ('Body Text', 'First Paragraph', 'Compact'):
        if name in [s.name for s in styles]:
            set_font(styles[name], size=11)
    for name in ('Table Caption', 'Image Caption', 'Caption'):
        if name in [s.name for s in styles]:
            set_font(styles[name], size=9, italic=True)

    # the "Table" table style
    tbl_style = None
    for s in styles:
        if s.name == 'Table':
            tbl_style = s
            break
    assert tbl_style is not None, 'pandoc Table style missing'
    se = tbl_style.element
    # 9 pt text for the whole table
    rpr = se.find(qn('w:rPr'))
    if rpr is None:
        rpr = el('rPr')
        se.append(rpr)
    for old in rpr.findall(qn('w:sz')) + rpr.findall(qn('w:szCs')):
        rpr.remove(old)
    rpr.append(el('sz', val=18))
    rpr.append(el('szCs', val=18))
    rfonts = el('rFonts', ascii='Times New Roman', hAnsi='Times New Roman')
    rpr.append(rfonts)
    # borders: thin top/bottom only; no verticals; padding 2 pt (40 twips)
    tblpr = se.find(qn('w:tblPr'))
    if tblpr is None:
        tblpr = el('tblPr')
        se.insert(0, tblpr)
    for old in tblpr.findall(qn('w:tblBorders')):
        tblpr.remove(old)
    borders = el('tblBorders')
    for side, v in [('top', 'single'), ('bottom', 'single'),
                    ('left', 'none'), ('right', 'none'),
                    ('insideH', 'none'), ('insideV', 'none')]:
        b = el(side, val=v)
        if v == 'single':
            b.set(qn('w:sz'), '4')
            b.set(qn('w:color'), '000000')
        borders.append(b)
    tblpr.append(borders)
    for old in tblpr.findall(qn('w:tblCellMar')):
        tblpr.remove(old)
    mar = el('tblCellMar')
    for side in ('top', 'bottom', 'left', 'right'):
        m = el(side, w=40, type='dxa')
        mar.append(m)
    tblpr.append(mar)
    # rule under the header row + bold header, via firstRow conditional style
    for old in se.findall(qn('w:tblStylePr')):
        if old.get(qn('w:type')) == 'firstRow':
            se.remove(old)
    first = el('tblStylePr', type='firstRow')
    frpr = el('rPr')
    frpr.append(el('b'))
    first.append(frpr)
    ftcpr = el('tcPr')
    fb = el('tcBorders')
    bb = el('bottom', val='single')
    bb.set(qn('w:sz'), '4')
    bb.set(qn('w:color'), '000000')
    fb.append(bb)
    ftcpr.append(fb)
    first.append(ftcpr)
    se.append(first)

    doc.save(str(OUT))
    print('reference.docx written:', OUT)


if __name__ == '__main__':
    main()
