import fitz
import numpy as np
import io
from PIL import Image

class PDF_reader:
    def __init__(self) -> None:
        self.reader = fitz
    
    def read_pdf(self, pdf_path:str, zoom:int = 100) -> dict:
        doc = self.reader.open(pdf_path)
        pages_data = {}
        for pg in range(doc.page_count):
            page = doc.load_page(pg)
            rotate = int(180)
            zoom = int(zoom)
            mat = fitz.Matrix(zoom/100.0, zoom/100.0).prerotate(rotate)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            page_data = []
            for i in range(pix.width):
                temp = []
                for j in range(pix.height):
                    r, g, b= pix.pixel(i, j)
                    temp.append([r,g,b])
                page_data.append(temp)
            pages_data[pg] = np.array(page_data)
        return pages_data
    
    def merge(self, data:dict, output:str):
        doc = fitz.Document(filetype="pdf")
        for k,v in data.items():
            image = Image.fromarray(v.astype("uint8"))
            image = image.rotate(-90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img_byte=io.BytesIO()
            image.save(img_byte,format='JPEG')
            img_byte=img_byte.getvalue()
            img = fitz.Document(stream=img_byte, filetype='jpg')
            imgPdf = fitz.Document("pdf", img.convert_to_pdf())
            doc.insert_pdf(imgPdf)
        doc.save(output)
        doc.close()
    