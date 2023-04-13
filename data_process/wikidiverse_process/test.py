from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
import os

path = "E:\HIRWorks\data\ImgData\\036d2c2342f620ce7df42fafbf58bb43.svg"

drawing = svg2rlg(path)
renderPM.drawToFile(drawing, "Pic.png")