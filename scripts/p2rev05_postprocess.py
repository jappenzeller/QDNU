#!/usr/bin/env python3
"""PROMPT P2REV-05 Job 4b - post-process manuscript_revised.docx.

- Tables 4, 5 and S1 in one landscape section; everything else portrait.
- Autofit off everywhere; explicit column widths proportional to content
  with a floor so no header wraps to more than two lines.
- Figures scaled to 16 cm width (aspect preserved); captions are below
  (pandoc implicit figures).
"""
from pathlib import Path
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Twips

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / 'PolarityPaper' / 'revision' / 'manuscript_revised.docx'

PAGE_W, PAGE_H = 12240, 15840          # letter, twips
MARGIN = 1080                          # 0.75 in
PORTRAIT_USABLE = PAGE_W - 2 * MARGIN
LANDSCAPE_USABLE = PAGE_H - 2 * MARGIN


def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(f'w:{k}'), str(v))
    return e


def sect_pr(landscape):
    s = el('sectPr')
    w, h = (PAGE_H, PAGE_W) if landscape else (PAGE_W, PAGE_H)
    pg = el('pgSz', w=w, h=h)
    if landscape:
        pg.set(qn('w:orient'), 'landscape')
    s.append(pg)
    s.append(el('pgMar', top=MARGIN, right=MARGIN, bottom=MARGIN,
                left=MARGIN, header=720, footer=720, gutter=0))
    return s


def para_with_sect(body, ref_el, landscape, before=True):
    p = el('p')
    ppr = el('pPr')
    ppr.append(sect_pr(landscape))
    p.append(ppr)
    if before:
        ref_el.addprevious(p)
    else:
        ref_el.addnext(p)


def set_col_widths(table, usable):
    tbl = table._tbl
    tblpr = tbl.tblPr
    for old in tblpr.findall(qn('w:tblLayout')):
        tblpr.remove(old)
    tblpr.append(el('tblLayout', type='fixed'))
    for old in tblpr.findall(qn('w:tblW')):
        tblpr.remove(old)
    tblpr.append(el('tblW', w=usable, type='dxa'))
    n_rows = len(table.rows)
    n_cols = len(table.columns)
    # weight per column: header length wrappable to two lines, plus the
    # longest body cell (capped) - crude but keeps headers on <= 2 lines
    weights = []
    for c in range(n_cols):
        header = table.rows[0].cells[c].text
        words = header.split()
        two_line = max((len(w) for w in words), default=4)
        two_line = max(two_line, -(-len(header) // 2))
        body = max((len(table.rows[r].cells[c].text)
                    for r in range(1, n_rows)), default=4)
        weights.append(max(two_line, min(body, 26), 4))
    total = sum(weights)
    widths = [max(int(usable * w / total), 500) for w in weights]
    widths[-1] = usable - sum(widths[:-1])
    grid = tbl.find(qn('w:tblGrid'))
    if grid is not None:
        for gc, w in zip(grid.findall(qn('w:gridCol')), widths):
            gc.set(qn('w:w'), str(w))
    for r in range(n_rows):
        for c, w in zip(range(n_cols), widths):
            tc = table.rows[r].cells[c]._tc
            tcpr = tc.get_or_add_tcPr()
            for old in tcpr.findall(qn('w:tcW')):
                tcpr.remove(old)
            tcpr.append(el('tcW', w=w, type='dxa'))


def main():
    doc = Document(str(DOC))
    body = doc.element.body

    # locate caption paragraphs for the landscape block
    paras = body.findall(qn('w:p'))
    cap4 = capS1_table = None
    for p in paras:
        text = ''.join(t.text or '' for t in p.iter(qn('w:t')))
        if text.startswith('Table 4.') and cap4 is None:
            cap4 = p
        if text.startswith('Table S1.'):
            capS1 = p
    assert cap4 is not None, 'Table 4 caption not found'
    # table following the S1 caption
    node = capS1
    while node is not None and node.tag != qn('w:tbl'):
        node = node.getnext()
    assert node is not None, 'Table S1 body not found'
    tblS1 = node

    # portrait section ends just before Table 4's caption;
    # landscape section ends just after Table S1
    para_with_sect(body, cap4, landscape=False, before=True)
    para_with_sect(body, tblS1, landscape=True, before=False)

    # column widths: landscape for tables 4,5,S1, portrait for the rest
    landscape_idx = set()
    all_tables = body.findall(qn('w:tbl'))
    pos4 = list(body).index(cap4)
    for i, tb in enumerate(all_tables):
        landscape_idx.add(i) if list(body).index(tb) > pos4 and \
            list(body).index(tb) <= list(body).index(tblS1) else None
    for i, table in enumerate(doc.tables):
        usable = LANDSCAPE_USABLE if i in landscape_idx else PORTRAIT_USABLE
        set_col_widths(table, usable)

    # figures to 16 cm
    for shape in doc.inline_shapes:
        if shape.width:
            ratio = shape.height / shape.width
            shape.width = Cm(16)
            shape.height = int(Cm(16) * ratio)

    doc.save(str(DOC))
    print('post-processed:', DOC.name, '| tables:', len(doc.tables),
          '| landscape tables:', sorted(landscape_idx),
          '| figures resized:', len(doc.inline_shapes))


if __name__ == '__main__':
    main()
