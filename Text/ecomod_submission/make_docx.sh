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
