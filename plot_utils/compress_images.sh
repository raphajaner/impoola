for pdf in procgen/ppo/nature/*.pdf; do
    gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH \
       -sOutputFile="$(basename "$pdf")" "$pdf"
done