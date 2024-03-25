from .extend import ImageBase, DTC_compressor, SVD_compressor, PCA_compressor
from .pdfreader import PDF_reader
import os

class PDF_compressor:
    def __init__(self, pdf_path:str, output:str, zoom:int) -> None:
        if not os.path.exists(pdf_path):
            raise("unexsist path")
        self.output = output
        self.reader = PDF_reader()
        self.data  = self.reader.read_pdf(pdf_path, zoom)
        self.compressor = {
            "dtc": DTC_compressor(),
            "svd": SVD_compressor(),
            "pca": PCA_compressor(),
        }
    
    def compress(self, method:str, k:int = 10):
        if method == None:
            raise ValueError("Method is not specified")
        if method not in self.compressor.keys():
            raise ValueError(f"{method} is not supported")
        for j,v in self.data.items():
            model = self.compressor[method]
            model.k = k
            self.data[j] = model.compress(v)
        self.reader.merge(self.data, self.output)