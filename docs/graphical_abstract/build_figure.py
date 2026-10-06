#!/usr/bin/env python3
"""Rebuild the illustrated SPATHI-benchmarks FULL graphical abstract.

This is a programmatic reconstruction, NOT the latent/original source of the
AI-generated reference PNG. The reference is never loaded, traced, embedded,
resized or copied by this program. Every visible element is constructed here.

Usage:
    python build_figure.py
    python build_figure.py --out-dir /tmp/figure --dpi 600
    python build_figure.py --font-dir /path/to/dejavu-fonts

Python >=3.10; dependencies are listed in requirements.txt. No Graphviz,
Inkscape, LaTeX, network connection, API, or generative model is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from contextlib import AbstractContextManager, contextmanager
from html import escape
from pathlib import Path
from typing import Any, Iterator, Sequence

import fitz
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent


def blend(a: str, b: str, t: float) -> str:
    return '#' + ''.join(f'{round(int(a[i:i+2],16)*(1-t)+int(b[i:i+2],16)*t):02X}'
                         for i in (1, 3, 5))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_font(filename: str, family: str, bold: bool, font_dir: Path | None) -> Path:
    """Resolve the exact requested font; do not silently substitute a family."""
    candidates = []
    if font_dir:
        candidates.append(font_dir / filename)
    candidates += [
        Path('/usr/share/fonts/truetype/dejavu') / filename,
        Path('/usr/local/share/fonts') / filename,
        Path.home() / '.fonts' / filename,
        Path.home() / '.local/share/fonts' / filename,
        Path('/Library/Fonts') / filename,
        Path.home() / 'Library/Fonts' / filename,
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / filename,
    ]
    if shutil.which('fc-match'):
        proc = subprocess.run(['fc-match', '-f', '%{file}',
                               f'{family}:style={"Bold" if bold else "Book"}'],
                              check=False, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout:
            candidates.append(Path(proc.stdout.strip()))
    for path in candidates:
        if path.is_file() and path.name.lower() == filename.lower():
            return path.resolve()
    raise RuntimeError(f'Missing font {filename}. Install DejaVu Sans Condensed, '
                       'or pass --font-dir pointing to its TTF files. '
                       'No font files are distributed in this package.')


def attr(**kwargs: Any) -> str:
    return ' '.join(f'{k.replace("xml_space", "xml:space").replace("_", "-")}="{escape(str(v), quote=True)}"'
                    for k, v in kwargs.items() if v is not None)


class Figure:
    """Write the same primitives to SVG and an actual vector PDF.

    Coordinates are top-left-origin design units. Fonts are subset-embedded
    in the PDF. The editable SVG retains text, matching SPATHI's graphical
    abstract and keeping one canonical SVG representation.
    """
    def __init__(self, cfg: dict[str, Any], out: Path, fonts: dict[bool, Path]):
        self.cfg, self.out, self.fonts = cfg, out, fonts
        self.c = cfg['palette']
        self.w, self.h = cfg['canvas']['width'], cfg['canvas']['height']
        self.scale = cfg['canvas']['physical_width_mm'] * 72 / 25.4 / self.w
        self.stem = cfg['output_basename']
        self.pdf = canvas.Canvas(str(out / f'{self.stem}.pdf'),
                                 pagesize=(self.w*self.scale, self.h*self.scale),
                                 pageCompression=1, invariant=1)
        self.pdf.setTitle('SPATHI-benchmarks (FULL) - reproducible vector reconstruction')
        self.pdf.setAuthor('Adrian Segura Ortiz; programmatic figure reconstruction')
        self.pdf.setSubject(cfg['provenance'])
        self.pdf.translate(0, self.h*self.scale)
        self.pdf.scale(self.scale, -self.scale)
        self.parts: list[str] = []
        self.defs: list[str] = []
        self.texts: list[dict[str, Any]] = []
        self.regions: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.ox = self.oy = 0.0
        self.primitive_count = 0
        for bold, path in fonts.items():
            pdfmetrics.registerFont(TTFont(self.font_name(bold), str(path)))
        self.rect(0, 0, self.w, self.h, self.c['white'])

    @staticmethod
    def font_name(bold: bool) -> str:
        return 'FigureBold' if bold else 'FigureRegular'

    @contextmanager
    def group(self, identifier: str, x: float = 0, y: float = 0,
              region: tuple[float, float, float, float] | None = None) -> Iterator[None]:
        old = self.current
        if region:
            xx, yy, w, h = region
            self.current = dict(id=identifier, x=xx+self.ox+x, y=yy+self.oy+y,
                                width=w, height=h)
            self.regions.append(self.current)
        self.parts.append(f'<g id="{escape(identifier)}" transform="translate({x:g},{y:g})">')
        self.pdf.saveState()
        self.pdf.translate(x, y)
        self.ox += x
        self.oy += y
        try:
            yield
        finally:
            self.ox -= x
            self.oy -= y
            self.pdf.restoreState()
            self.parts.append('</g>')
            self.current = old

    def _paint(self, fill: str | None, stroke: str | None, sw: float) -> None:
        if fill and fill != 'none': self.pdf.setFillColor(HexColor(fill))
        if stroke and stroke != 'none': self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.setLineWidth(sw)
        self.pdf.setLineCap(1)
        self.pdf.setLineJoin(1)

    def rect(self, x: float, y: float, w: float, h: float, fill: str | Sequence[str] = 'none',
             stroke: str | None = None, r: float = 0, sw: float = 1,
             dash: bool = False) -> None:
        self.primitive_count += 1
        gradient = not isinstance(fill, str)
        fill_string = str(fill)
        if gradient:
            g = f'gradient-{len(self.defs)}'
            self.defs.append(f'<linearGradient id="{g}" x1="0" y1="0" x2="1" y2="1">'
                             f'<stop offset="0" stop-color="{fill[0]}"/>'
                             f'<stop offset="1" stop-color="{fill[1]}"/></linearGradient>')
            fill_string = f'url(#{g})'
        self.parts.append('<rect '+attr(x=x,y=y,width=w,height=h,rx=r,fill=fill_string,
                                       stroke=stroke,stroke_width=sw,
                                       stroke_dasharray='4 3' if dash else None) + '/>')
        self.pdf.saveState()
        self.pdf.setDash([4,3] if dash else [])
        if gradient:
            self.pdf.saveState()
            p = self.pdf.beginPath()
            p.roundRect(x,y,w,h,r)
            self.pdf.clipPath(p, stroke=0, fill=0)
            self.pdf.linearGradient(x,y,x+w,y+h,[HexColor(fill[0]),HexColor(fill[1])])
            self.pdf.restoreState()
        self._paint(None if gradient else fill,stroke,sw)
        self.pdf.roundRect(x,y,w,h,r,stroke=int(bool(stroke)),
                           fill=int(not gradient and fill!='none'))
        self.pdf.restoreState()

    def circle(self,x:float,y:float,r:float,fill:str='none',stroke:str|None=None,sw:float=1,
               dash:bool=False) -> None:
        self.primitive_count += 1
        self.parts.append('<circle '+attr(cx=x,cy=y,r=r,fill=fill,stroke=stroke,
                                         stroke_width=sw,stroke_dasharray='4 3' if dash else None)+'/>' )
        self.pdf.saveState()
        self._paint(fill,stroke,sw)
        self.pdf.setDash([4,3] if dash else [])
        self.pdf.circle(x,y,r,stroke=int(bool(stroke)),fill=int(fill!='none'))
        self.pdf.restoreState()

    def ellipse(self,x:float,y:float,rx:float,ry:float,fill:str='none',stroke:str|None=None,
                sw:float=1) -> None:
        self.primitive_count += 1
        self.parts.append('<ellipse '+attr(cx=x,cy=y,rx=rx,ry=ry,fill=fill,
                                          stroke=stroke,stroke_width=sw)+'/>' )
        self.pdf.saveState(); self._paint(fill,stroke,sw)
        self.pdf.ellipse(x-rx,y-ry,x+rx,y+ry,stroke=int(bool(stroke)),fill=int(fill!='none'))
        self.pdf.restoreState()

    def path(self,d:str,stroke:str|None=None,sw:float=1.5,fill:str='none',dash:bool=False) -> None:
        """Absolute SVG path commands M, L, C, Q and Z are supported."""
        self.primitive_count += 1
        self.parts.append('<path '+attr(d=d,stroke=stroke,stroke_width=sw,fill=fill,
                            stroke_linecap='round',stroke_linejoin='round',
                            stroke_dasharray='4 3' if dash else None)+'/>' )
        tokens = re.findall(r'[MLCQZmlcqz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', d)
        p = self.pdf.beginPath(); i=0; current=(0.,0.)
        while i < len(tokens):
            op=tokens[i]; i+=1
            n={'M':2,'L':2,'C':6,'Q':4,'Z':0}.get(op)
            if n is None: raise ValueError(f'Unsupported path operation {op!r}')
            a=list(map(float,tokens[i:i+n])); i+=n
            if op=='M': p.moveTo(*a); current=tuple(a)
            elif op=='L': p.lineTo(*a); current=tuple(a)
            elif op=='C': p.curveTo(*a); current=tuple(a[-2:])
            elif op=='Q':
                x0,y0=current; x1,y1,x2,y2=a
                p.curveTo(x0+2*(x1-x0)/3,y0+2*(y1-y0)/3,
                          x2+2*(x1-x2)/3,y2+2*(y1-y2)/3,x2,y2)
                current=(x2,y2)
            else: p.close()
        self.pdf.saveState(); self._paint(fill,stroke,sw)
        self.pdf.setDash([4,3] if dash else [])
        self.pdf.drawPath(p,stroke=int(bool(stroke)),fill=int(fill!='none'))
        self.pdf.restoreState()

    def line(self,x1:float,y1:float,x2:float,y2:float,color:str,sw:float=1.5) -> None:
        self.path(f'M {x1} {y1} L {x2} {y2}',color,sw)

    def arrow(self,x1:float,y1:float,x2:float,y2:float,color:str|None=None,
              sw:float=1.7,head:float=6) -> None:
        color=color or self.c['muted']
        a=math.atan2(y2-y1,x2-x1)
        bx=x2-head*math.cos(a); by=y2-head*math.sin(a)
        self.line(x1,y1,bx,by,color,sw)
        pts=[(x2,y2),(bx-head*.58*math.sin(a),by+head*.58*math.cos(a)),
             (bx+head*.58*math.sin(a),by-head*.58*math.cos(a))]
        self.path('M '+' L '.join(f'{x:g} {y:g}' for x,y in pts)+' Z',None,0,color)

    def text(self,x:float,y:float,value:str|Sequence[str],size:float=14,
             bold:bool=False,color:str|None=None,anchor:str='start',
             maxw:float|None=None,leading:float|None=None) -> None:
        lines=[value] if isinstance(value,str) else value
        for j,s in enumerate(lines):
            yy=y+j*(leading or size*1.25)
            width=pdfmetrics.stringWidth(str(s),self.font_name(bold),size)
            sx=min(1.0,maxw/width) if maxw and width else 1.0
            visual_w=width*sx
            xx=x-(visual_w/2 if anchor=='middle' else visual_w if anchor=='end' else 0)
            self.parts.append('<text '+attr(x=0,y=0,
                transform=f'translate({xx:g},{yy:g}) scale({sx:.8f},1)',
                font_family=self.cfg['font_family'],font_size=size,
                font_weight='bold' if bold else 'normal',
                fill=color or self.c['ink'],xml_space='preserve')+'>'+escape(str(s))+'</text>')
            self.pdf.saveState(); self.pdf.translate(xx,yy); self.pdf.scale(sx,-1)
            self.pdf.setFont(self.font_name(bold),size)
            self.pdf.setFillColor(HexColor(color or self.c['ink']))
            self.pdf.drawString(0,0,str(s)); self.pdf.restoreState()
            ascent,descent=pdfmetrics.getAscentDescent(self.font_name(bold),size)
            self.texts.append(dict(text=str(s),x=xx+self.ox,y=yy-ascent+self.oy,
                            width=visual_w,height=ascent-descent,size=size,
                            horizontal_scale=sx,region=self.current['id'] if self.current else None))

    def finish(self,dpi:int) -> dict[str,Any]:
        self.pdf.showPage(); self.pdf.save()
        svg=('<?xml version="1.0" encoding="UTF-8"?>\n'
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
             f'viewBox="0 0 {self.w} {self.h}">\n'
             '<title>SPATHI-benchmarks (FULL)</title>\n'
             f'<desc>{escape(self.cfg["provenance"])}</desc>\n'
             '<defs>'+''.join(self.defs)+'</defs>\n'+ '\n'.join(self.parts)+'\n</svg>\n')
        (self.out/f'{self.stem}.svg').write_text(svg,encoding='utf-8')
        with fitz.open(self.out/f'{self.stem}.pdf') as doc:
            page=doc[0]
            page.get_pixmap(dpi=dpi,alpha=False).save(self.out/f'{self.stem}.png')
            image_count=len(page.get_images(full=True))
            drawing_count=len(page.get_drawings())
            text_span_count=sum(len(line['spans']) for b in page.get_text('dict')['blocks']
                                if 'lines' in b for line in b['lines'])
        regions={r['id']:r for r in self.regions}
        outside=[]
        for t in self.texts:
            r=regions.get(t['region'],dict(x=0,y=0,width=self.w,height=self.h))
            if (t['x']<r['x']-1 or t['y']<r['y']-1 or
                t['x']+t['width']>r['x']+r['width']+1 or
                t['y']+t['height']>r['y']+r['height']+1):
                outside.append(t)
        overlaps=[]
        for i,a in enumerate(self.texts):
            for b in self.texts[i+1:]:
                w=min(a['x']+a['width'],b['x']+b['width'])-max(a['x'],b['x'])
                h=min(a['y']+a['height'],b['y']+b['height'])-max(a['y'],b['y'])
                if w>1 and h>1:
                    overlaps.append(dict(a=a['text'],b=b['text'],intersection=[w,h]))
        checks=dict(text_elements=len(self.texts),vector_primitives=self.primitive_count,
                    pdf_drawing_objects=drawing_count,pdf_text_spans=text_span_count,
                    raster_images_in_pdf=image_count,text_outside_regions=outside,
                    text_box_overlaps=overlaps,reference_image_used_by_builder=False,
                    bbox_note='Font ascent/descent boxes; no claim of a complete geometry proof.')
        (self.out/'layout_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
        (self.out/'text_geometry.json').write_text(json.dumps(self.texts,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        return checks


# ---------------- All icons are original, deterministic vector primitives ----------------
def database(f:Figure,x:float,y:float,s:float,color:str) -> None:
    f.rect(x-s*.38,y-s*.25,s*.76,s*.8,color,r=s*.07)
    for dy in [.50,.23,-.03,-.29]:
        f.ellipse(x,y+dy*s,s*.39,s*.145,color,'#FFFFFF',1.5)
        if dy!=-.29:
            f.rect(x-s*.375,y+(dy-.20)*s,s*.75,s*.16,color)
    f.ellipse(x,y-s*.29,s*.39,s*.145,color,'#FFFFFF',1.5)


def gear(f:Figure,x:float,y:float,s:float,color:str) -> None:
    points=[]
    for k in range(12):
        for da,r in [(-.40,.37),(-.30,.49),(.30,.49),(.40,.37)]:
            a=2*math.pi*(k+da)/12
            points.append((x+s*r*math.cos(a),y+s*r*math.sin(a)))
    f.path('M '+' L '.join(f'{a:g} {b:g}' for a,b in points)+' Z',color,.5,color)
    f.circle(x,y,s*.19,'#FFFFFF')


def document(f:Figure,x:float,y:float,w:float,h:float,color:str) -> None:
    d=f'M {x+5} {y} L {x+w*.70} {y} L {x+w} {y+h*.23} L {x+w} {y+h-4} Q {x+w} {y+h} {x+w-4} {y+h} L {x+4} {y+h} Q {x} {y+h} {x} {y+h-4} L {x} {y+4} Q {x} {y} {x+5} {y} Z'
    f.path(d,color,2.4,'#FFFFFF')
    f.path(f'M {x+w*.70} {y} L {x+w*.70} {y+h*.24} L {x+w} {y+h*.24}',color,2)
    for t in [.45,.62,.79]: f.line(x+w*.25,y+h*t,x+w*.70,y+h*t,color,2.7)


def small_network(f:Figure,x:float,y:float,s:float,color:str,branch:bool=False) -> None:
    if branch:
        pts=[(0,-.39),(-.24,-.06),(.23,-.06),(-.44,.36),(-.09,.34),(.40,.34)]
        edges=[(0,1),(0,2),(1,3),(1,4),(2,5)]
    else:
        pts=[(.18,-.33),(-.28,.02),(.18,.37)]
        edges=[(0,1),(1,2),(2,0)]
    for a,b in edges:
        ax,ay=pts[a]; bx,by=pts[b]; f.line(x+ax*s,y+ay*s,x+bx*s,y+by*s,color,1.8)
    for ax,ay in pts: f.circle(x+ax*s,y+ay*s,s*.11,color,blend(color,'#FFFFFF',.2),.6)


def bars(f:Figure,x:float,y:float,w:float,h:float,color:str) -> None:
    for i,t in enumerate([.40,.67,1]):
        f.rect(x+i*w*.34,y+h*(1-t),w*.24,h*t,blend(color,'#FFFFFF',.22-.07*i),r=.7)


def people(f:Figure,x:float,y:float,s:float,color:str) -> None:
    for dx in [-s*.20,s*.20]:
        f.circle(x+dx,y-s*.23,s*.14,color)
        f.path(f'M {x+dx-s*.18} {y+s*.30} Q {x+dx-s*.20} {y-s*.05} {x+dx} {y-s*.04} Q {x+dx+s*.20} {y-s*.05} {x+dx+s*.18} {y+s*.30} Z',None,0,color)


def tree(f:Figure,x:float,y:float,w:float,h:float,color:str,depth:int=3) -> None:
    """Compact branching symbol, matching the illustrative reference."""
    pts=[(x+w/2,y)]
    parents=[0]
    all_edges=[]
    for level in range(1,depth+1):
        count=2**level
        # Narrow internal levels, broad final level.
        spread=w*(level/depth)
        new=[]
        for k in range(count):
            px=x+w/2-spread/2+spread*(k+.5)/count
            py=y+h*level/depth
            idx=len(pts); pts.append((px,py)); new.append(idx)
            all_edges.append((parents[k//2],idx))
        parents=new
    for a,b in all_edges: f.line(*pts[a],*pts[b],color,max(.85,w/32))
    for px,py in pts: f.circle(px,py,max(1.65,min(2.8,w/12)),color)


def dataset(f:Figure,x:float,y:float,w:float,h:float,color:str,label:str|Sequence[str]|None=None,
            mixed:bool=False,seed:int=0) -> None:
    f.rect(x,y,w,h,['#FFFFFF','#F8FCFD'],f.c['line'],4,1)
    top=y+10
    if label:
        f.text(x+w/2,y+16,label,10.2,True,anchor='middle',maxw=w-8,leading=10.8)
        top=y+(33 if not isinstance(label,str) else 30)
    # The cloud is an illustration, not a plotted experimental dataset.
    pts=[(.22,.31),(.37,.12),(.54,.23),(.69,.39),(.46,.48),(.28,.61),(.65,.69),(.40,.83),(.81,.65),(.17,.83),(.82,.17),(.60,.93)]
    colors=[color, f.c['orange'],f.c['teal'],f.c['purple']]
    for k,(a,b) in enumerate(pts):
        kk=(k+seed)%len(pts)
        a,b=pts[kk]
        px=x+8+a*(w-16); py=top+b*(h-(top-y)-10)
        co=colors[(k+seed)%4] if mixed else blend(color,'#FFFFFF',.13*(k%3))
        f.circle(px,py,2.4 if w<55 else 2.8,co,'#FFFFFF',.45)


def grid(f:Figure,x:float,y:float,cols:int,rows:int,cell:float,color:str,light:bool=False) -> None:
    for j in range(rows):
        for i in range(cols):
            tone=.48 if light else .10
            tone+=.08*((i*3+j*2)%4)
            f.rect(x+i*cell,y+j*cell,cell-1.4,cell-1.4,
                   blend(color,'#FFFFFF',tone),blend(color,'#FFFFFF',.70),.6,.5)


def trophy(f:Figure,x:float,y:float,s:float,color:str) -> None:
    f.path(f'M {x-s*.24} {y-s*.35} L {x+s*.24} {y-s*.35} L {x+s*.20} {y+s*.04} Q {x} {y+s*.31} {x-s*.20} {y+s*.04} Z',color,1.2,blend(color,'#FFFFFF',.08))
    for sign in [-1,1]:
        f.path(f'M {x+sign*s*.24} {y-s*.29} L {x+sign*s*.38} {y-s*.29} Q {x+sign*s*.40} {y+s*.05} {x+sign*s*.19} {y+s*.04}',color,2)
    f.line(x,y+s*.12,x,y+s*.37,color,3)
    f.rect(x-s*.20,y+s*.35,s*.40,s*.09,color,r=1)


def snowflake(f:Figure,x:float,y:float,s:float,color:str) -> None:
    for k in range(6):
        a=k*math.pi/3
        x2=x+s*.46*math.cos(a); y2=y+s*.46*math.sin(a)
        f.line(x,y,x2,y2,color,2)
        bx=x+s*.29*math.cos(a); by=y+s*.29*math.sin(a)
        for sign in [-1,1]:
            aa=a+sign*math.pi/3
            f.line(bx,by,bx+s*.14*math.cos(aa),by+s*.14*math.sin(aa),color,1.8)


def check(f:Figure,x:float,y:float,s:float,color:str) -> None:
    f.path(f'M {x-s*.35} {y} L {x-s*.07} {y+s*.28} L {x+s*.38} {y-s*.32}',color,2.2)


def cpu(f:Figure,x:float,y:float,s:float,color:str) -> None:
    f.rect(x-s*.31,y-s*.31,s*.62,s*.62,color,r=2)
    f.rect(x-s*.21,y-s*.21,s*.42,s*.42,'#FFFFFF',r=1)
    f.rect(x-s*.15,y-s*.15,s*.30,s*.30,color,r=1)
    for k in range(4):
        off=(-.24+k*.16)*s
        for sign in [-1,1]:
            f.line(x+off,y+sign*s*.32,x+off,y+sign*s*.46,color,2.5)
            f.line(x+sign*s*.32,y+off,x+sign*s*.46,y+off,color,2.5)


def memory(f:Figure,x:float,y:float,w:float,h:float,color:str) -> None:
    f.rect(x,y,w,h,color,r=3)
    for k in range(4): f.rect(x+w*(.14+k*.20),y+h*.25,w*.10,h*.45,'#FFFFFF',r=1)
    for k in range(7): f.line(x+w*(.08+k*.14),y+h-1,x+w*(.08+k*.14),y+h+3,color,2)


def stopwatch(f:Figure,x:float,y:float,s:float,color:str) -> None:
    f.circle(x,y,s*.32,'#FFFFFF',color,3)
    f.line(x,y,x,y-s*.20,color,2)
    f.line(x,y,x+s*.14,y+s*.08,color,2)
    f.rect(x-s*.12,y-s*.52,s*.24,s*.09,color,r=1)
    f.line(x,y-s*.43,x,y-s*.32,color,2)
    f.line(x+s*.28,y-s*.29,x+s*.34,y-s*.35,color,3)


def hourglass(f:Figure,x:float,y:float,s:float,color:str) -> None:
    f.line(x-s*.25,y-s*.45,x+s*.25,y-s*.45,color,2)
    f.line(x-s*.25,y+s*.45,x+s*.25,y+s*.45,color,2)
    f.path(f'M {x-s*.20} {y-s*.42} Q {x-s*.20} {y-s*.13} {x} {y} Q {x-s*.20} {y+s*.13} {x-s*.20} {y+s*.42} L {x+s*.20} {y+s*.42} Q {x+s*.20} {y+s*.13} {x} {y} Q {x+s*.20} {y-s*.13} {x+s*.20} {y-s*.42} Z',color,1.6,blend(color,'#FFFFFF',.87))


def link(f:Figure,x:float,y:float,s:float,color:str) -> None:
    # Two oblique open chain links.
    f.path(f'M {x-s*.05} {y-s*.15} L {x+s*.16} {y-s*.36} C {x+s*.38} {y-s*.58} {x+s*.65} {y-s*.30} {x+s*.43} {y-s*.08} L {x+s*.20} {y+s*.13}',color,2.5)
    f.path(f'M {x+s*.02} {y+s*.14} L {x-s*.17} {y+s*.34} C {x-s*.39} {y+s*.56} {x-s*.63} {y+s*.29} {x-s*.41} {y+s*.06} L {x-s*.18} {y-s*.16}',color,2.5)
    f.line(x-s*.15,y+s*.17,x+s*.18,y-s*.17,color,2.5)


def dna(f:Figure,x:float,y:float,s:float,color:str) -> None:
    for sign in [-1,1]:
        f.path(f'M {x+sign*s*.24} {y-s*.45} C {x+sign*s*.28} {y-s*.05} {x-sign*s*.25} {y-s*.05} {x-sign*s*.24} {y+s*.43}',color,2.2)
    for t in [-.32,-.18,.02,.19,.32]:
        width=abs(math.sin(t*math.pi*2))*s*.22
        f.line(x-width,y+t*s,x+width,y+t*s,color,1.4)


def target(f:Figure,x:float,y:float,s:float,color:str) -> None:
    for r in [.40,.26,.12]: f.circle(x,y,s*r,'none',color,2)
    f.arrow(x,y,x+s*.40,y-s*.46,color,2,5)


def panel(f:Figure,key:str,data:dict[str,Any],number:int,color:str) -> None:
    p=f.cfg['panels'][key]; x,y,w,h=[p[k] for k in ['x','y','width','height']]
    f.rect(x,y,w,h,'#FFFFFF',color,6,2.2)
    f.rect(x+1,y+1,w-2,66,[blend(color,'#000000',.08),blend(color,'#FFFFFF',.07)],r=4)
    f.rect(x+1,y+60,w-2,7,color)
    f.circle(x+40,y+31,25,'#FFFFFF')
    f.text(x+40,y+42,str(number),31,True,color,'middle')
    with f.group(key+'-header',region=(x,y,w,66)):
        # The two title blocks are deliberately bounded, reproducing the reference.
        titlew=215 if key=='comparison' else 188
        f.text(x+79,y+41,data['title'],25,True,'#FFFFFF',maxw=titlew)
        tagline_x=x+(305 if key=='comparison' else 278)
        f.text(tagline_x,y+27,data['tagline'],13.4,False,'#FFFFFF',
               maxw=x+w-16-tagline_x,leading=16.3)


def step(f:Figure,identifier:str,x:float,y:float,w:float,h:float,color:str,
         letter:str,title:str|Sequence[str],desc:str|Sequence[str],
         title_size:float=15.2,desc_size:float=13.3) -> AbstractContextManager[None]:
    # Used as context manager via step_context below.
    return step_context(f,identifier,x,y,w,h,color,letter,title,desc,title_size,desc_size)


@contextmanager
def step_context(f:Figure,identifier:str,x:float,y:float,w:float,h:float,color:str,
                 letter:str,title:str|Sequence[str],desc:str|Sequence[str],
                 title_size:float,desc_size:float) -> Iterator[None]:
    with f.group(identifier, x, y, (0,0,w,h)):
        f.rect(0,0,w,h,[blend(color,'#FFFFFF',.945),blend(color,'#FFFFFF',.975)],r=6)
        f.circle(23,24,21,blend(color,'#FFFFFF',.80))
        f.circle(23,24,16.4,color,'#FFFFFF',1.2)
        f.text(23,30.1,letter,20,True,'#FFFFFF','middle')
        f.text(57,20,title,title_size,True,maxw=w-64,leading=18)
        n=1 if isinstance(title,str) else len(title)
        f.text(57,20+18*n+2,desc,desc_size,color=f.c['muted'],maxw=w-65,leading=16)
        yield


def count_card(f:Figure,x:float,y:float,w:float,h:float,lines:Sequence[str],color:str) -> None:
    f.rect(x,y,w,h,[blend(color,'#FFFFFF',.91),blend(color,'#FFFFFF',.96)],r=5)
    f.text(x+w/2,y+20,lines[0],15.1,True,color,'middle')
    f.text(x+w/2,y+36,list(lines[1:]),12.6,color=f.c['muted'],anchor='middle',
           maxw=w-10,leading=15)


def draw_calibration(f:Figure,t:dict[str,Any]) -> None:
    co=f.c['teal']; panel(f,'calibration',t,1,co)
    x=32; w=443
    with step(f,'calibration-a',x,302,w,168,co,'A',t['a_title'],t['a_desc']):
        for j,c in enumerate([co,f.c['orange'],f.c['purple']]):
            dataset(f,13+j*73,75,62,78,c,['Dataset 1','Dataset 2','Dataset …'][j],mixed=j==1,seed=j)
        f.text(237,119,'…',17,True,f.c['muted'],'middle')
        f.line(258,72,258,158,f.c['line'],1)
        small_network(f,283,86,25,'#285D77',True)
        bars(f,274,105,20,23,'#285D77')
        people(f,284,146,29,'#285D77')
        for k,label in enumerate(t['a_labels']):
            f.text(308,90+k*30,label,12.0,color=f.c['muted'],maxw=132)
    f.arrow(243,472,243,489)
    with step(f,'calibration-b',x,490,w,147,co,'B',t['b_title'],t['b_desc']):
        f.rect(7,53,324,85,'#FFFFFF',r=5)
        f.rect(337,53,99,79,blend(co,'#FFFFFF',.94),r=5)
        for j in range(3):
            for i in range(4): f.rect(23+i*26,62+j*19,21,15,[blend(co,'#FFFFFF',.86),blend(co,'#FFFFFF',.70)],blend(co,'#FFFFFF',.48),2,.8)
        f.text(140,87,'…',19,True,f.c['muted'],'middle')
        f.arrow(155,91,176,91,head=7)
        for j in range(3): tree(f,180+j*48,74,45,34,co,3)
        f.text(82,132,t['b_grid'],12.5,color=f.c['muted'],anchor='middle')
        f.text(254,131,t['b_forest'],12.5,color=f.c['muted'],anchor='middle')
        f.text(386,80,t['b_count'][0],15,True,co,'middle')
        f.text(386,96,t['b_count'][1:],13,color=f.c['muted'],anchor='middle',leading=16)
    f.arrow(243,639,243,656)
    with step(f,'calibration-c',x,657,w,185,co,'C',t['c_title'],t['c_desc'],desc_size=12.8):
        f.rect(5,53,330,125,'#FFFFFF',r=5)
        f.rect(52,61,87,91,blend(co,'#FFFFFF',.945),r=4)
        trophy(f,29,108,39,'#DB8907')
        for j in range(4):
            check(f,66,76+j*20,8,f.c['ink'])
            f.text(80,80+j*20,f'Finalist {j+1}',11.5,color=f.c['muted'])
        f.line(154,55,154,167,f.c['line'],1)
        for j,trees in enumerate([25,50,100,250]):
            hh=[13,22,32,40][j]; xx=166+j*43
            tree(f,xx,100-hh,25,hh,co,min(3,j+1))
            f.text(xx+12.5,120,str(trees),11.8,True,f.c['muted'],'middle')
        f.path('M 168 130 L 168 135 L 316 135 L 316 130',f.c['muted'],1)
        f.text(242,154,t['c_bracket'],11.2,color=f.c['muted'],anchor='middle',leading=14.3)
        for j,lines in enumerate(t['c_counts']): count_card(f,342,54+j*60,94,55,lines,co)
    f.arrow(243,844,243,855)
    with step(f,'calibration-d',x,856,w,139,co,'D',t['d_title'],t['d_desc'],desc_size=12.1):
        f.rect(6,56,222,76,'#FFFFFF',r=5)
        document(f,20,72,41,46,'#375B72')
        f.text(76,83,'·',13,True,color=f.c['muted'])
        f.text(92,83,t['d_notes'][0],11.0,color=f.c['muted'])
        f.text(76,101,'·',13,True,color=f.c['muted'])
        f.text(92,101,t['d_notes'][1:],11.0,color=f.c['muted'],leading=14)
        f.rect(241,53,195,79,'#FFFFFF','#8CCAF0',8,1)
        f.rect(249,61,179,63,['#EBF7FF','#F6FBFF'],r=4)
        snowflake(f,273,86,33,f.c['blue'])
        f.text(306,81,t['d_freeze'],13.4,True,maxw=113)
        f.text(306,100,t['d_freeze_desc'],10.7,color=f.c['muted'],maxw=117,leading=14)


def draw_comparison(f:Figure,t:dict[str,Any]) -> None:
    co=f.c['orange']; panel(f,'comparison',t,2,co)
    x=514; w=421
    with step(f,'comparison-a',x,302,w,168,co,'A',t['a_title'],t['a_desc'],desc_size=13.0):
        f.rect(12,57,208,100,'#FFFFFF',r=4)
        f.rect(241,57,168,100,'#FFFFFF',r=4)
        f.text(116,74,t['a_primary'],11.4,True,anchor='middle',maxw=194)
        f.text(325,74,t['a_deployment'],11.4,True,anchor='middle',maxw=162)
        for j,c in enumerate([f.c['teal'],f.c['purple'],f.c['purple']]):
            dataset(f,24+j*49,89,44,56,c,seed=j)
            f.rect(19+j*49,84,44,56,'none',f.c['line'],4,.7)
        f.text(202,124,'…',16,True,anchor='middle')
        for j,c in enumerate([co,'#167CE7']):
            f.rect(247+j*78,83,66,56,'none',f.c['line'],4,.7)
            dataset(f,254+j*78,91,66,58,c,seed=j+2)
    f.arrow(718,472,718,489)
    with step(f,'comparison-b',x,490,w,185,co,'B',t['b_title'],t['b_desc']):
        dataset(f,26,86,87,82,co,['Test','dataset'],mixed=True,seed=1)
        colors=[f.c['teal'],co,f.c['blue'],f.c['purple']]
        for j,(s,c) in enumerate(zip(t['b_methods'],colors)):
            yy=69+j*26
            f.rect(221,yy,161,22,[blend(c,'#FFFFFF',.93),'#FFFFFF'],c,4,.85)
            f.text(301.5,yy+15,s,11.1,True,c,'middle',maxw=151)
            f.path(f'M 124 {108+j*9} L 168 {yy+11} L 207 {yy+11}',f.c['muted'],1.5)
            f.arrow(204,yy+11,214,yy+11,head=5)
        f.text(165,175,'…',17,True,color=f.c['muted'],anchor='middle')
    f.arrow(718,677,718,692)
    with step(f,'comparison-c',x,694,w,164,co,'C',t['c_title'],t['c_desc'],desc_size=12.8):
        colors=['#008A7A',co,f.c['blue'],f.c['purple'],'#C72546']
        for j,(lines,c) in enumerate(zip(t['c_families'],colors)):
            xx=9+j*81
            f.rect(xx,50,75,82,[blend(c,'#FFFFFF',.965),blend(c,'#FFFFFF',.93)],blend(c,'#FFFFFF',.70),5,.9)
            cx=xx+37.5
            if j==0: document(f,cx-8,60,16,23,c)
            elif j==1: link(f,cx,72,28,c)
            elif j==2: small_network(f,cx,72,33,c,True)
            elif j==3: dna(f,cx,72,32,c)
            else:
                pts=[(cx,64),(cx-10,75),(cx+10,75),(cx-18,84),(cx,84),(cx+18,84)]
                for k,(px,py) in enumerate(pts):
                    if k: f.line(cx,64,px,py,blend(c,'#FFFFFF',.60),.7)
                    f.circle(px,py,3.3,c,'#FFFFFF',.5)
            f.text(cx,101,lines,11.3,True,c,'middle',maxw=70,leading=12.2)
        f.rect(5,139,w-10,22,blend(f.c['purple'],'#FFFFFF',.85),r=5)
        f.text(w/2,154,t['c_requests'],12.2,True,f.c['purple_dark'],'middle')
    f.arrow(718,860,718,872)
    with step(f,'comparison-d',x,874,w,121,co,'D',t['d_title'],t['d_desc'],desc_size=12.7):
        colors=['#008575',f.c['blue'],f.c['purple']]
        widths=[66,70,90]; starts=[11,86,165]
        for xx,ww,c in zip(starts,widths,colors):
            f.rect(xx,51,ww,64,[blend(c,'#FFFFFF',.94),'#FFFFFF'],blend(c,'#FFFFFF',.40),5,.9)
        bars(f,31,61,26,25,colors[0]); f.text(44,104,'AUPR',12.5,True,colors[0],'middle')
        f.path('M 102 61 L 102 89 L 141 89',colors[1],1.4)
        f.path('M 104 85 C 109 71 114 62 140 62',colors[1],2)
        f.text(121,105,'AUROC',11.8,True,colors[1],'middle')
        target(f,211,70,26,colors[2]); f.text(210,98,['F1 at','truth count'],11.6,True,colors[2],'middle',leading=13)
        f.line(266,54,266,116,f.c['line'],1)
        bars(f,278,58,23,25,'#47718D')
        f.text(315,65,t['d_stats'],11.1,color=f.c['muted'],maxw=102,leading=15)


def draw_scalability(f:Figure,t:dict[str,Any]) -> None:
    co=f.c['purple']; panel(f,'scalability',t,3,co)
    x=976; w=440
    with step(f,'scalability-a',x,302,w,198,co,'A',t['a_title'],t['a_desc'],desc_size=13.8):
        for j,(xx,cols,rows,cs) in enumerate([(19,4,3,12),(127,4,4,13),(238,5,4,13),(354,5,5,14)]):
            hh=rows*cs; yy=138-hh
            grid(f,xx,yy,cols,rows,cs,co if j>1 else '#6383B6',j<2)
            f.text(xx+cols*cs/2,154,t['a_sizes'][j],12.5,color=f.c['muted'],anchor='middle')
            if j<3: f.arrow(xx+cols*cs+17,119,xx+cols*cs+40,119,f.c['muted'],1.8,7)
        f.rect(6,167,w-12,26,blend(co,'#FFFFFF',.86),r=6)
        f.text(w/2,185,t['a_count'],12.2,True,co,'middle')
    f.arrow(1194,502,1194,514)
    with step(f,'scalability-b',x,515,w,110,co,'B',t['b_title'],t['b_desc']):
        cpu(f,110,77,45,'#386780')
        f.text(142,73,['CPU','(controlled)'],12.5,color=f.c['muted'],leading=15)
        f.line(228,54,228,99,'#A8B9D2',1.3)
        memory(f,254,61,47,26,'#205B70')
        f.text(315,73,['RAM','(controlled)'],12.5,color=f.c['muted'],leading=15)
    f.arrow(1194,627,1194,640)
    with step(f,'scalability-c',x,642,w,158,co,'C',t['c_title'],t['c_desc']):
        f.rect(9,61,71,50,'#FFFFFF',f.c['line'],4,1)
        grid(f,20,70,5,4,9,co,True)
        f.text(44,128,['Medium','dataset'],12.4,color=f.c['muted'],anchor='middle',leading=15)
        colors=[f.c['teal'],'#F09A1E',co]
        ys=[65,96,131]
        for j,(yy,c,lines) in enumerate(zip(ys,colors,t['c_cpu'])):
            f.arrow(92,89+j*10,157,yy,c if j==0 else f.c['muted'],1.5,6)
            f.circle(175,yy,11.3,blend(c,'#FFFFFF',.56),c,1)
            f.text(201,yy+5,lines,12.7,color=f.c['muted'],maxw=102,leading=14)
        f.rect(311,51,118,92,'#FFFFFF',blend(co,'#FFFFFF',.43),6,1,True)
        f.text(370,76,t['c_reuse'],11.8,color=f.c['muted'],anchor='middle',leading=16,maxw=106)
    f.arrow(1194,802,1194,816)
    with step(f,'scalability-d',x,817,w,178,co,'D',t['d_title'],t['d_desc'],desc_size=13.1):
        f.rect(7,49,w-14,79,'#FFFFFF',r=6)
        xs=[47,122,211,288,379]
        stopwatch(f,xs[0],72,30,f.c['teal_dark'])
        memory(f,xs[1]-18,65,36,16,f.c['purple_dark'])
        f.circle(xs[2],71,14.5,f.c['green']); check(f,xs[2],71,17,'#FFFFFF')
        f.circle(xs[3],71,14.5,f.c['red'])
        f.line(xs[3]-5,66,xs[3]+5,76,'#FFFFFF',2.6)
        f.line(xs[3]+5,66,xs[3]-5,76,'#FFFFFF',2.6)
        f.circle(xs[4],71,14,'none',f.c['gray'],2,True)
        f.line(xs[4]-7,79,xs[4]+7,63,f.c['gray'],2)
        for xx,lines in zip(xs,t['d_status']):
            f.text(xx,100,lines,11.7,color=f.c['muted'],anchor='middle',maxw=85,leading=14)
        f.rect(7,133,153,40,blend(co,'#FFFFFF',.88),r=5)
        hourglass(f,25,153,27,co)
        f.text(46,148,t['d_limit'],10.8,color=co,maxw=108,leading=14)
        f.rect(165,133,268,40,blend(co,'#FFFFFF',.88),r=5)
        f.circle(181,153,12,co); f.line(175,153,187,153,'#FFFFFF',2.4)
        f.text(201,146,t['d_stop'],10.4,color=co,maxw=225,leading=11.5)


def draw_title_footer(f:Figure,t:dict[str,Any]) -> None:
    with f.group('title',region=(20,0,1415,119)):
        f.text(30,65,t['title'],61.5,True,maxw=857)
        f.text(38,103,t['subtitle'],24.0,True,f.c['muted'],maxw=932)
        f.line(1007,30,1007,100,'#ABBBD0',1.5)
        f.text(1037,61,t['title_note'],19.2,color=f.c['muted'],leading=24,maxw=266)
        # Same visual mark as the reference: a branching network plus bars.
        pts=[(1360,27),(1341,55),(1378,55),(1325,80),(1355,80)]
        for a,b in [(0,1),(0,2),(1,3),(1,4)]: f.line(*pts[a],*pts[b],'#37617A',1.9)
        for k,(xx,yy) in enumerate(pts): f.circle(xx,yy,8.8 if k==0 else 7.6,f.c['teal'] if k==0 else '#4F7D93','#E3F3F7',.9)
        bars(f,1378,47,40,39,'#249CAD')
        f.text(1371,112,'S P A T H I',16.8,True,anchor='middle',maxw=124)
    cards=[(20,350),(382,337),(731,337),(1080,350)]
    colors=[f.c['teal'],f.c['teal'],f.c['orange'],f.c['purple']]
    for i,((x,w),co,lines) in enumerate(zip(cards,colors,t['summary'])):
        with f.group(f'summary-{i}',region=(x,125,w,90)):
            f.rect(x,125,w,90,[blend(co,'#FFFFFF',.92),blend(co,'#FFFFFF',.96)],r=8)
            if i==0: database(f,x+48,168,55,co)
            elif i==1: gear(f,x+48,169,51,co)
            elif i==2: document(f,x+27,146,41,51,co)
            else: small_network(f,x+46,170,54,co)
            f.text(x+94,175,lines[0],48,True,co)
            f.text(x+96,199,lines[1],17,color=f.c['muted'],maxw=w-103)
    # The three workflow panels are sequential; these arrows match the reference.
    for xx in [480,943]:
        f.path(f'M {xx} 588 L {xx+15} 588 L {xx+15} 581 L {xx+28} 594 L {xx+15} 607 L {xx+15} 600 L {xx} 600 Z',None,0,'#245F7C')
    with f.group('footer',region=(20,1020,1410,45)):
        ft=t['footer']
        f.rect(20,1020,1410,45,'#FFFFFF',f.c['line'],6,1.1)
        database(f,197,1042,32,'#3D6C89')
        f.text(229,1040,ft['repro_title'],14.5,True)
        f.text(229,1056,ft['repro_text'],10.6,color=f.c['muted'],maxw=490)
        f.line(728,1029,728,1056,f.c['line'],1)
        document(f,839,1029,19,28,'#416F8D')
        f.text(889,1040,ft['output_title'],14.5,True)
        f.text(889,1056,ft['output_text'],10.7,color=f.c['muted'])
    f.text(20,1080,t['footer']['note'],8.5,color=f.c['muted'])
    f.text(1424,1080,t['footer']['signature'],8.5,color=f.c['muted'],anchor='end')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'figure_config.json')
    parser.add_argument('--text',type=Path,default=ROOT/'figure_text.json')
    parser.add_argument('--out-dir',type=Path,default=ROOT)
    parser.add_argument('--font-dir',type=Path)
    parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args()
    try:
        if not 72<=args.dpi<=1200: raise ValueError('--dpi must be between 72 and 1200')
        cfg=json.loads(args.config.read_text(encoding='utf-8'))
        text=json.loads(args.text.read_text(encoding='utf-8'))
        args.out_dir.mkdir(parents=True,exist_ok=True)
        fonts={bold:find_font(cfg['font_files']['bold' if bold else 'regular'],
                             cfg['font_family'],bold,args.font_dir) for bold in [False,True]}
        f=Figure(cfg,args.out_dir,fonts)
        draw_calibration(f,text['calibration'])
        draw_comparison(f,text['comparison'])
        draw_scalability(f,text['scalability'])
        draw_title_footer(f,text)
        checks=f.finish(args.dpi)
        import reportlab
        env=dict(python=sys.version.split()[0],reportlab=reportlab.Version,pymupdf=fitz.VersionBind,
                 fonts={('bold' if b else 'regular'):dict(filename=p.name,sha256=sha256(p))
                        for b,p in fonts.items()},
                 canvas=cfg['canvas'],dpi=args.dpi,
                 inputs={args.config.name:sha256(args.config),args.text.name:sha256(args.text),
                         Path(__file__).name:sha256(Path(__file__))},
                 protocol_commit=cfg['protocol_commit'],reference_image_used=False)
        (args.out_dir/'build_environment.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(checks,indent=2))
        if checks['raster_images_in_pdf'] or checks['text_outside_regions'] or checks['text_box_overlaps']:
            print('Layout validation failed; inspect layout_checks.json.',file=sys.stderr)
            return 2
        print(f'Wrote SVG, PDF and PNG files to {args.out_dir.resolve()}')
        return 0
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print(f'Figure build failed: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
