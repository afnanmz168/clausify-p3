#!/bin/sh
# Rebuild the LaTeX + PDF of PROJECT_REPORT.md. Run from anywhere.
set -e
cd "$(dirname "$0")"
sed -e '1d' -e '3,5d' ../PROJECT_REPORT.md > body.md   # title/authors go in the LaTeX title block (meta.yaml)
pandoc body.md -f markdown-auto_identifiers -t latex -s \
  --metadata-file=meta.yaml --columns=80 -o PROJECT_REPORT.tex
rm body.md
tectonic -X compile PROJECT_REPORT.tex
