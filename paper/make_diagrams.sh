#!/bin/zsh
# Compile the TikZ diagrams in figures_src/ to vector PDFs in figures/ (run from paper2_fodm/).
set -e
cd "${0:A:h}/figures_src"
for f in fig_framework fig_designspace; do
  pdflatex -interaction=nonstopmode -halt-on-error $f.tex > $f.build.log 2>&1 || { echo "FAILED $f (see figures_src/$f.build.log)"; exit 1; }
  cp $f.pdf ../figures/$f.pdf
  echo "wrote figures/$f.pdf"
done
rm -f *.aux
