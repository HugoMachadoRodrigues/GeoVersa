#!/bin/zsh
# Build manuscript.docx from the same LaTeX source as the PDF.
# --lua-filter strips the bookmarks and internal hyperlinks pandoc would
# otherwise attach to every heading and cross-reference.
cd "$(dirname "$0")"
cat > /tmp/nolinks.lua <<'LUA'
-- drop internal links, keep their text
function Link(el)
  if el.target:sub(1,1) == "#" then return el.content end
  return el
end
-- drop anchors/spans that become Word bookmarks
function Span(el)
  if el.identifier and el.identifier ~= "" then el.identifier = "" end
  return el
end
function Header(el)
  el.identifier = ""
  return el
end
LUA
# pandoc does not know the macros this document defines, and silently drops
# their contents: \file{} emptied every script name and every profile label,
# turning a list of six profiles into ", , , , , .". The \@eadauthor line, which
# exists to keep the ORCID mark out of the PDF's e-mail footnote, leaked into the
# Word text as "=Hugo Rodrigues". Both are handled by rewriting a copy of the
# source before conversion; the PDF still builds from the untouched original.
python3 - <<'PYPRE'
import re
s = open("manuscript.tex").read()
s = re.sub(r'\\makeatletter\\@eadauthor=\{[^}]*\}\\makeatother\s*', '', s)
def esc(m):
    return "\\texttt{" + m.group(1).replace("_", "\\_") + "}"
s = re.sub(r'\\file\{([^}]*)\}', esc, s)
open("/tmp/manuscript_docx.tex", "w").write(s)
print("  source pre-processed for pandoc")
PYPRE
cp -f references.bib elsevier-harv-expanded.csl /tmp/ 2>/dev/null
pandoc /tmp/manuscript_docx.tex \
  --bibliography=references.bib --citeproc \
  --csl=elsevier-harv-expanded.csl \
  --lua-filter=/tmp/nolinks.lua \
  --resource-path=.:figures \
  -o manuscript.docx

# pandoc still emits bookmarks for captions and reference entries; strip every
# remaining bookmark and internal anchor directly from the document XML.
python3 - <<'PY'
import zipfile, re, shutil, os
src = "manuscript.docx"; tmp = "/tmp/_docx_clean.docx"
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        x = data.decode("utf8")
        x = re.sub(r'<w:bookmarkStart[^>]*/>', '', x)
        x = re.sub(r'<w:bookmarkEnd[^>]*/>', '', x)
        data = x.encode("utf8")
    zout.writestr(item, data)
zout.close(); zin.close(); shutil.move(tmp, src)
PY

# elsarticle's \author/\address/\ead macros are not understood by pandoc, so the
# affiliation, e-mail and keywords never reach the Word file. Insert them as real
# paragraphs immediately after the title.
python3 - <<'PYX'
import zipfile, re, shutil
src="manuscript.docx"; tmp="/tmp/_docx_meta.docx"
NS='xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
def para(text, italic=False, bold=False):
    rpr = "<w:rPr>" + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + "</w:rPr>"
    return ('<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r>' + rpr +
            '<w:t xml:space="preserve">' + text + '</w:t></w:r></w:p>')
block = (para("Hugo Rodrigues" + chr(0x1D43) + ", Golmar Golmohammadi", bold=True) +
         para("University of Florida, Range Cattle Research and Education Center, "
              "3401 Experiment Station, Ona, FL 33865, USA", italic=True) +
         para("ORCID: Hugo Rodrigues 0000-0002-8070-8126, "
              "Golmar Golmohammadi 0000-0001-5532-3892", italic=True) +
         para(chr(0x1D43) + " Corresponding author: rodrigues.machado.hugo@gmail.com",
              italic=True) +
         '<w:p/>')
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename=="word/document.xml":
        x=d.decode("utf8")
        # pandoc renders the \author macros itself now that the \makeatletter
        # line is pre-processed away, so its bare Author paragraphs would sit
        # alongside the formatted block below. Drop them and keep ours, which
        # carries the affiliation, both ORCIDs and the corresponding author.
        x = re.sub(r'<w:p>\s*<w:pPr>\s*<w:pStyle w:val="Author"\s*/>\s*</w:pPr>.*?</w:p>',
                   '', x, flags=re.S)
        if "Range Cattle Research" not in x:
            m=re.search(r'(<w:body>)', x)
            if m: x = x[:m.end()] + block + x[m.end():]
        # pandoc ignores elsarticle's \sep, so the keyword list arrives as one
        # run of comma-jammed terms with no label. Rebuild that paragraph.
        tex=open("manuscript.tex").read()
        kwblock=re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', tex, re.S)
        KW=[k.strip() for k in kwblock.group(1).replace("\n"," ").split("\\sep") if k.strip()]
        i=x.find(KW[0])
        if i>=0:
            s=x.rfind("<w:p>",0,i); e=x.find("</w:p>",i)+len("</w:p>")
            kwp=('<w:p><w:r><w:rPr><w:b/></w:rPr>'
                 '<w:t xml:space="preserve">Keywords: </w:t></w:r>'
                 '<w:r><w:t xml:space="preserve">' + "; ".join(KW) + '</w:t></w:r></w:p>')
            x=x[:s]+kwp+x[e:]
            print("  keyword list rebuilt")
        d=x.encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
print("  affiliation block inserted")
PYX

# pandoc drops the caption numbering that elsarticle generates, and emits the
# bibliography with no heading. Number the captions in document order and insert
# a References heading before the first bibliography entry.
python3 - <<'PYY'
import zipfile, re, shutil
src="manuscript.docx"; tmp="/tmp/_docx_caps.docx"
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename=="word/document.xml":
        x=d.decode("utf8")
        counters={"ImageCaption":[0,"Figure"], "TableCaption":[0,"Table"]}
        def number(m):
            sty=m.group(1); head=m.group(0)
            counters[sty][0]+=1
            lbl="%s %d. " % (counters[sty][1], counters[sty][0])
            # prefix the first <w:t> that follows this paragraph properties block
            return head + '<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % lbl
        x=re.sub(r'<w:pPr>\s*<w:pStyle w:val="(ImageCaption|TableCaption)"\s*/>\s*</w:pPr>',
                 number, x)
        # References heading before the first Bibliography paragraph
        # The first paragraph carrying the Bibliography style IS the insertion
        # point. The earlier version searched backwards from it for another
        # "<w:p>", which landed one paragraph too early and dropped the heading
        # into the middle of the data-availability statement.
        bi = x.find('w:pStyle w:val="Bibliography"')
        start = x.rfind("<w:p>", 0, bi) if bi > 0 else -1
        if start >= 0 and "References</w:t>" not in x:
            head=('<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
                  '<w:r><w:t xml:space="preserve">References</w:t></w:r></w:p>')
            x=x[:start]+head+x[start:]
        d=x.encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
print("  captions numbered, References heading added")
PYY

# Long tables split across pages. Word cannot keep a whole table together, but it
# can stop a single row breaking mid-row and repeat the header on the next page,
# which turns a broken table into a readable one.
# pandoc drops the section numbering elsarticle generates, so the manuscript
# arrived in Word with unnumbered headings while its own prose sent the reader
# to "Section 7.4". It also renders \paragraph as Heading4, skipping level 3,
# and gives the appendix no label. All three are fixed here from document order
# rather than from a hardcoded list, so a new section does not silently break
# the count.
python3 - <<'PYN'
import zipfile, re, shutil
src="manuscript.docx"; tmp="/tmp/_docx_num.docx"
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename=="word/document.xml":
        x=d.decode("utf8")
        paras=list(re.finditer(r"<w:p[ >].*?</w:p>", x, re.S))
        h1=0; h2=0; lastlvl=1; appendix=False; out=[]; last=0
        # everything from the appendix heading onward stops taking a number,
        # and the declarations that follow it are unnumbered in the source too
        for m in paras:
            body=m.group(0)
            lvl=re.search(r'w:val="Heading(\d)"', body)
            if not lvl: continue
            n=int(lvl.group(1)); txt=re.sub(r"<[^>]+>","",body)
            pre=None
            if n==1:
                if txt.startswith("Full computational specification"):
                    appendix=True; pre="Appendix A. "
                elif not appendix:
                    h1+=1; h2=0; pre=f"{h1}. "
            elif n==2 and not appendix:
                h2+=1; pre=f"{h1}.{h2}. "
            new=body
            if n <= 2:
                lastlvl = n
            else:
                # A \\paragraph sits one level under whatever section contains
                # it. Pinning it at 3 left a 1 -> 3 jump wherever a paragraph
                # follows a section directly, which is every first paragraph
                # in the appendix and the one in the Discussion.
                tgt = min(lastlvl + 1, 3)
                new = new.replace('w:val="Heading%d"' % n, 'w:val="Heading%d"' % tgt)
            if pre:
                new=re.sub(r"(<w:t(?: [^>]*)?>)", lambda q: q.group(1)+pre, new, count=1)
            if new!=body:
                out.append((m.start(), m.end(), new))
        buf=[]
        for a,b,new in out:
            buf.append(x[last:a]); buf.append(new); last=b
        buf.append(x[last:])
        x="".join(buf)
        d=x.encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
print("  headings numbered, appendix labelled, level-4 headings demoted to 3")
PYN

# pandoc gives every column of every table the same width: the text block
# divided by the column count. A ten-column table then gets 0.55 cm per column
# and Word breaks "Simple" across three lines. Widths are reassigned here in
# proportion to the longest cell in each column, and the layout set to autofit
# so Word may still adjust.
python3 - <<'PYW'
import zipfile, re, shutil
src="manuscript.docx"; tmp="/tmp/_docx_w.docx"
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
TOTAL=9360    # twips in the portrait text block
WIDE_TOTAL=12960  # and in the landscape one, for the two tables placed there
WIDE=("DesignIntervalPICP", "DesignUnits/pts", "EstimandReps")
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename=="word/document.xml":
        x=d.decode("utf8")
        def fix(m):
            tb=m.group(0)
            rows=re.findall(r"<w:tr[ >].*?</w:tr>", tb, re.S)
            if not rows: return tb
            # A spanning row, the italic subheading a \\multicolumn produces, has
            # fewer cells than the table has columns and its whole text would
            # otherwise be charged to the first column. Width is measured only
            # from rows that carry a cell for every column.
            per_row=[re.findall(r"<w:tc>.*?</w:tc>", r, re.S) for r in rows]
            ncol=max((len(c) for c in per_row), default=0)
            widths=[0]*ncol
            for cells in per_row:
                if len(cells)!=ncol: continue
                for i,c in enumerate(cells):
                    widths[i]=max(widths[i], len(re.sub(r"<[^>]+>","",c)))
            n=len(widths)
            if n==0: return tb
            hdr=re.sub(r"<[^>]+>","", rows[0])
            total = WIDE_TOTAL if any(hdr.startswith(w) for w in WIDE) else TOTAL
            # Width from what the cell has to hold, not from its share of the
            # character count. A pure proportion starved the sample-size column,
            # whose longest entry is three digits, until Word broke "200" after
            # the second digit. CHAR is a conservative advance width at the
            # table's type size and PAD covers the cell margins on both sides.
            CHAR, PAD = 115, 300
            need=[w*CHAR + PAD for w in widths]
            tot=sum(need)
            if tot <= total:
                # It fits, so leave it at the width it needs. Distributing the
                # slack in proportion handed the widest column most of it and
                # stretched a six-centimetre text column to fourteen.
                cols=list(need)
            else:
                cols=[max(int(total*nw/tot), 620) for nw in need]
            gi=iter(cols)
            tb=re.sub(r'<w:gridCol w:w="\d+"', lambda q: '<w:gridCol w:w="%d"'%next(gi, 800), tb)
            if "<w:tblLayout" in tb:
                tb=re.sub(r'<w:tblLayout w:type="\w+"/>', '<w:tblLayout w:type="autofit"/>', tb)
            else:
                tb=tb.replace("</w:tblPr>", '<w:tblLayout w:type="autofit"/></w:tblPr>', 1)
            return tb
        x=re.sub(r"<w:tbl>.*?</w:tbl>", fix, x, flags=re.S)
        d=x.encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
print("  table column widths set in proportion to content")
PYW

# pandoc maps LaTeX \varrho to U+1D71A, MATHEMATICAL ITALIC RHO SYMBOL, which
# lives outside the Basic Multilingual Plane. Word substitutes it from Cambria
# Math, but LibreOffice and several PDF viewers have no glyph and draw a red
# box, which is what a reader checking Eq. 11 sees. The plain Greek rho carries
# the same meaning, is present in every font, and is already italicised by the
# surrounding OMML run. Any other astral-plane character would fail the same
# way, so the whole Mathematical Alphanumeric Symbols block is folded down.
python3 - <<'PYU'
import zipfile, re, shutil, unicodedata
src="manuscript.docx"; tmp="/tmp/_docx_u.docx"
def fold(ch):
    o=ord(ch)
    if not (0x1D400 <= o <= 0x1D7FF): return ch
    n=unicodedata.name(ch, "")
    for greek in ("ALPHA","BETA","GAMMA","DELTA","EPSILON","ZETA","ETA","THETA",
                  "IOTA","KAPPA","LAMDA","MU","NU","XI","OMICRON","PI","RHO",
                  "SIGMA","TAU","UPSILON","PHI","CHI","PSI","OMEGA"):
        if greek in n:
            small = "SMALL" in n or "ITALIC" in n and "CAPITAL" not in n
            try:
                return unicodedata.lookup("GREEK %s LETTER %s" % ("SMALL" if small else "CAPITAL", greek))
            except KeyError:
                return ch
    d=unicodedata.decomposition(ch)
    if d and d.split()[0].startswith("<"):
        try: return chr(int(d.split()[-1], 16))
        except Exception: return ch
    return ch
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
n_fold=0
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename.endswith(".xml"):
        x=d.decode("utf8")
        out=[]
        for ch in x:
            f=fold(ch)
            if f!=ch: n_fold+=1
            out.append(f)
        d="".join(out).encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
print("  %d astral-plane math characters folded to BMP equivalents" % n_fold)
PYU

# The coverage table and the sensitivity table are seven and ten columns wide.
# Proportional widths stopped the worst of the breaking but Word still stacks
# "n=200" and ".915" vertically in the ten-column one, and the reviewer was
# right that shrinking the font further is not the answer. Both are placed in
# their own landscape section, caption included, which is the ordinary
# typographic remedy for a table that does not fit the measure.
python3 - <<'PYL'
import zipfile, re, shutil
src="manuscript.docx"; tmp="/tmp/_docx_ls.docx"
PORT = ('<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"'
        ' w:header="720" w:footer="720" w:gutter="0"/>')
LAND = ('<w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"'
        ' w:header="720" w:footer="720" w:gutter="0"/>')
def brk(sect):
    return '<w:p><w:pPr><w:sectPr>%s</w:sectPr></w:pPr></w:p>' % sect
# a table is chosen by the text of its header row, so renumbering cannot
# silently move the landscape to the wrong one
WIDE = ("DesignIntervalPICP", "DesignUnits/pts", "EstimandReps")
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename=="word/document.xml":
        x=d.decode("utf8")
        n=0
        for tb in re.findall(r"<w:tbl>.*?</w:tbl>", x, re.S):
            hdr=re.sub(r"<[^>]+>","", re.search(r"<w:tr[ >].*?</w:tr>", tb, re.S).group(0))
            if not any(hdr.startswith(w) for w in WIDE): continue
            i=x.find(tb)
            # take the caption paragraph that precedes the table along with it
            pre=x.rfind("<w:p>", 0, i)
            pre2=x.rfind("<w:p ", 0, i)
            start=max(pre, pre2)
            if start < 0: start=i
            j=i+len(tb)
            x = x[:start] + brk(PORT) + x[start:j] + brk(LAND) + x[j:]
            n+=1
        d=x.encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
print("  %d wide tables placed in landscape sections" % n)
PYL

python3 - <<'PYZ'
import zipfile, re, shutil
src="manuscript.docx"; tmp="/tmp/_docx_tbl.docx"
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename=="word/document.xml":
        x=d.decode("utf8")
        # no row may split across a page
        x=x.replace("<w:trPr>", "<w:trPr><w:cantSplit/>")
        x=re.sub(r'<w:tr>(?!\s*<w:trPr>)', '<w:tr><w:trPr><w:cantSplit/></w:trPr>', x)
        # repeat the first row of every table as a header
        def hdr(m):
            body=m.group(0)
            i=body.find("<w:tr>")
            if i<0: return body
            j=body.find("</w:tr>", i)
            first=body[i:j]
            if "<w:tblHeader/>" not in first:
                first=first.replace("<w:trPr>", "<w:trPr><w:tblHeader/>", 1)
            return body[:i]+first+body[j:]
        x=re.sub(r'<w:tbl>.*?</w:tbl>', hdr, x, flags=re.S)
        # No keepNext inside the tables. It was added when the sensitivity
        # table sat in portrait and ended with one row alone on a page, and it
        # solved that. Once the wide tables moved to landscape they fit their
        # page with room to spare, and the binding stopped earning its cost:
        # combined with a bound caption it made caption and table one
        # indivisible block, so the coverage table took a page of its own and
        # left its caption on the page before. cantSplit still stops a single
        # row breaking across pages, and the header row still repeats.
        # Binding the rows made each table a rigid block, and a rigid block that
        # does not fit the space left after its caption moves to the next page
        # and leaves the caption behind. The caption has to travel with it, so
        # the paragraph immediately before each table gets keepNext too.
        out=[]; pos=0
        for m in re.finditer(r"<w:tbl>", x):
            i=m.start()
            a=max(x.rfind("<w:p>", pos, i), x.rfind("<w:p ", pos, i))
            if a < 0: continue
            b=x.find("</w:p>", a)
            if b < 0 or b > i: continue
            cap=x[a:b+6]
            if "<w:keepNext/>" in cap: continue
            if "<w:pPr>" in cap:
                cap2=cap.replace("<w:pPr>", "<w:pPr><w:keepNext/>", 1)
            else:
                cap2=re.sub(r"(<w:p[ >][^>]*>)", r"\1<w:pPr><w:keepNext/></w:pPr>", cap, count=1)
            out.append((a, b+6, cap2)); pos=i
        for a,b,cap2 in reversed(out):
            x = x[:a] + cap2 + x[b:]
        print("  %d captions bound to their table" % len(out))
        d=x.encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
print("  table rows set to not split; header rows repeat")
PYZ

# Word refuses the whole package when a custom document property exceeds the
# 255-character limit of vt:lpwstr. pandoc turns the LaTeX abstract into exactly
# such a property, at 1548 characters, which is why the file opened everywhere
# except in Word. The body was never the problem.
#
# The same pass drops the duplicate w:tblHeader this script used to add on top of
# the one pandoc already emits for header rows.
python3 - <<'PYW'
import zipfile, re, shutil
src="manuscript.docx"; tmp="/tmp/_docx_props.docx"
LIMIT = 255
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    d=zin.read(it.filename)
    if it.filename=="docProps/custom.xml":
        x=d.decode("utf8")
        def trim(m):
            head, typ, val, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            if typ=="lpwstr" and len(val)>LIMIT:
                val = val[:LIMIT-1].rsplit(" ",1)[0] + "\u2026"
            return head+val+tail
        x=re.sub(r'(<vt:(lpwstr)>)(.*?)(</vt:lpwstr>)',
                 lambda m: trim(re.match(r'(<vt:(lpwstr)>)(.*?)(</vt:lpwstr>)', m.group(0), re.S)),
                 x, flags=re.S)
        over=[len(v) for v in re.findall(r'<vt:lpwstr>(.*?)</vt:lpwstr>', x, re.S) if len(v)>LIMIT]
        print(f"  custom properties trimmed to {LIMIT} chars ({len(over)} still over)")
        d=x.encode("utf8")
    elif it.filename=="word/document.xml":
        x=d.decode("utf8")
        # keep pandoc's tblHeader, drop the one this script injected
        x=re.sub(r'<w:tblHeader/>(?=(?:(?!</w:trPr>).)*<w:tblHeader\b)', '', x, flags=re.S)
        d=x.encode("utf8")
    zout.writestr(it,d)
zout.close(); zin.close(); shutil.move(tmp,src)
PYW
