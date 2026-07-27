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
pandoc manuscript.tex \
  --bibliography=references.bib --citeproc \
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
