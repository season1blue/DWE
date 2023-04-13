from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

def svg2png(svgFile, pngFile):
    drawing = svg2rlg(svgFile)
    renderPM.drawToFile(drawing, pngFile)

def gif2png(gifFile, pngFile):
    im = Image.open(gifFile)
    im.tell()
    im.save(pngFile)
