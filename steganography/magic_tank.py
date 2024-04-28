import cv2
import numpy as np

def load_image(f:any):
    _ = np.frombuffer(f.bdata.hex, np.uint8)
    wm = cv2.imdecode(_, cv2.IMREAD_COLOR)
    return wm

def magic_tank(file1:any, file2:any, output:str, a=0.5, b=20):
    A = load_image(file1)
    B = load_image(file2)

    height, width, _ = A.shape
    B = cv2.resize(B, (width, height))

    A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    B_gray = a * B_gray + b
    alpha = 255 - A_gray.astype('int') + B_gray.astype('int')
    alpha = np.clip(alpha, 1, 255).reshape(height, width, 1)
    P = (A - (255 - alpha)) / (alpha / 255)
    alpha = alpha.astype('u8')
    P = np.clip(P, 0, 255)
    image_with_alpha = np.concatenate([P, alpha], axis=2)
    cv2.imwrite(output, image_with_alpha)