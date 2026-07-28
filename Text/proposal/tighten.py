# Pandoc's default Word geometry is too airy for a one-page brief. Pull the
# margins in and cut the space around headings and paragraphs.
import zipfile, re, shutil, sys
src = sys.argv[1] if len(sys.argv) > 1 else "proposal.docx"
tmp = "/tmp/_tight.docx"
zin = zipfile.ZipFile(src); zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    d = zin.read(it.filename)
    if it.filename == "word/document.xml":
        x = d.decode("utf8")
        x = re.sub(r'<w:pgMar[^/]*/>',
                   '<w:pgMar w:top="1010" w:right="1010" w:bottom="1010" '
                   'w:left="1010" w:header="360" w:footer="360" w:gutter="0"/>', x)
        d = x.encode("utf8")
    elif it.filename == "word/styles.xml":
        x = d.decode("utf8")
        x = re.sub(r'<w:spacing w:before="\d+"', '<w:spacing w:before="120"', x)
        x = re.sub(r'w:after="\d+" w:line="\d+" w:lineRule="auto"',
                   'w:after="60" w:line="240" w:lineRule="auto"', x)
        d = x.encode("utf8")
    zout.writestr(it, d)
zout.close(); zin.close(); shutil.move(tmp, src)
print("  margins and spacing tightened")
