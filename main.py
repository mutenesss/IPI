import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def inputData() -> list:
    imgname = str(input("Insira o nome da imagem a ser processada: "))
    imgcolor = int(input("Imagem colorida?\nSim=1, Nao=0: "))
    return [imgname, imgcolor]

def process_choice(color : int) -> int:
    print("Escolha o processo a ser executado:\n")
    print("1 - Diminuir linhas pela metade.")
    print("2 - Diminuir colunas pela metade.")
    print("3 - Diminuir linhas e colunas pela metade.")
    print("4 - Duplicar linhas.")
    print("5 - Duplicar colunas.")
    print("6 - Duplicar linhas e colunas(mono).")
    if color == 1:
        print("7 - Diminuir linhas e colunas(color).")
        print("8 - Filtro Laplaciano +-8.")
        print("9 - Filtro Gaussiano e Filtro Laplaciano +-4.")
        print("10 - Filtro Ideal vs Filtro Gaussiano.")
    else:
        print("7 - Filtro Laplaciano +-8.")
        print("8 - Filtro Gaussiano e Filtro Laplaciano +-4.")
        print("9 - Filtro Ideal vs Filtro Gaussiano.")
    val = int(input("Operacao desejada: "))
    return val



def half_rows(img: np.ndarray, color: int = 0, name: str = os.getcwd()+"/result/saida_linhas.png") -> np.ndarray:
    if color == 1:
        result = img[::2, :, :]
    else:
        result = img[::2, :]
    result = np.array(result)
    cv2.imwrite(os.getcwd+name, result)
    return result


def half_columns(img: np.ndarray, color: int = 0, name: str = os.getcwd()+"/result/saida_colunas.png") -> np.ndarray:
    if color == 1:
        result = img[:, ::2, :]
    else:
        result = img[:, ::2]
    result = np.array(result)
    cv2.imwrite(os.getcwd+name, result)
    return result


def quarter_size(img: np.ndarray, color: int = 0, name: str = os.getcwd()+"/result/saida_redux.png") -> np.ndarray:
    name = os.getcwd+name
    return half_columns(half_rows(img, color), color, name)


def double_rows(img: np.ndarray, name: str = None) -> np.ndarray:
    height = img.shape[0]
    img_norm = img.astype(np.float32) / 255.0
    result_norm = np.zeros((height*2,) + img_norm.shape[1:], dtype=np.float32)

    for i in range(height):
        result_norm[i*2] = img_norm[i]
        if i < height - 1:
            result_norm[i*2+1] = (img_norm[i] + img_norm[i+1]) / 2.0
        else:
            result_norm[i*2+1] = img_norm[i]
    result = (result_norm * 255.0).astype(np.uint8)
    if name:
        cv2.imwrite(os.getcwd()+name, result)
    return result


def double_cols(img: np.ndarray, name: str = None) -> np.ndarray:
    width = img.shape[1]
    img_norm = img.astype(np.float32) / 255.0
    result_norm = np.zeros((img_norm.shape[0], width*2) + img_norm.shape[2:], dtype=np.float32)
    for i in range(width):
        result_norm[:, i*2] = img_norm[:, i]
        if i < width - 1:
            result_norm[:, i*2+1] = (img_norm[:, i] + img_norm[:, i+1]) / 2.0
        else:
            result_norm[:, i*2+1] = img_norm[:, i]
    result = (result_norm * 255.0).astype(np.uint8)
    if name:
        cv2.imwrite(os.getcwd()+"/result/"+name, result)
    return result


def double_img(img: np.ndarray, name: str = None) -> np.ndarray:
    result = double_cols(double_rows(img))
    if name:
        cv2.imwrite(name, result)
    return result

def save_channels(b:np.ndarray, g:np.ndarray, r:np.ndarray, scale:bool=False) -> None:
    path = os.getcwd()
    height, width = b.shape
    zeros = np.zeros((height, width), dtype=np.uint8)
    b_color = np.stack([b, zeros, zeros], axis=2)   
    g_color = np.stack([zeros, g, zeros], axis=2)   
    r_color = np.stack([zeros, zeros, r], axis=2)   
    if scale:
        cv2.imwrite(path+"/result/b_up.png", b_color)
        cv2.imwrite(path+"/result/g_up.png", g_color)
        cv2.imwrite(path+"/result/r_up.png", r_color)
    else:
        cv2.imwrite(path+"/result/b_down.png", b_color)
        cv2.imwrite(path+"/result/g_down.png", g_color)
        cv2.imwrite(path+"/result/r_down.png", r_color)

def double_channels(img: np.ndarray, name: str = None) -> np.ndarray:
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    b_up = double_img(b)
    g_up = double_img(g)
    r_up = double_img(r)
    save_channels(b_up, g_up, r_up, scale=True)

    result = np.stack([b_up, g_up, r_up], axis=2)
    if name:
        cv2.imwrite(os.getcwd()+"/result/"+name, result)
    return result

def half_channels(img:np.ndarray, name:str=None) -> np.ndarray:
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    b_down = quarter_size(b)
    g_down = quarter_size(g)
    r_down = quarter_size(r)
    save_channels(b_down, g_down, r_down, scale=False)

    result = np.stack([b_down, g_down, r_down], axis=2)
    if name:
        cv2.imwrite(os.getcwd()+"/result/"+name, result)
    return result

def call_laplacian(img:np.ndarray, kernel_size:int = 3, name:str = None) -> list:
    realceLap = img.astype(np.float32)

    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]],dtype=np.float32)
    bordas_k = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel)
    re1 = img.astype(np.float32) - bordas_k.astype(np.float32)
    re1 = np.clip(re1, 0, 255)
    re1 = re1.astype(np.uint8)
    
    bordasLap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=kernel_size)
    re2 = realceLap - bordasLap
    re2 = np.clip(re2, 0,255)
    re2 = re2.astype(np.uint8)
    if name:
        cv2.imwrite(os.getcwd()+"/result/lap_"+name,re2)
        cv2.imwrite(os.getcwd()+"/result/ker3_"+name,re1)
        cv2.imwrite(os.getcwd()+"/result/lap_border_"+name,bordasLap)
        cv2.imwrite(os.getcwd()+"/result/ker3_border_"+name,bordas_k)
    return [re1, re2]

def call_gaus_lap(img: np.ndarray, kernel_size:int, var:float, name:str = None) -> np.ndarray:
    sigma = np.sqrt(var)
    blur = cv2.GaussianBlur(img, sigmaX=sigma,ksize=(kernel_size, kernel_size))
    kernel = np.array([[0,1,0], [1,-4,1],[0,1,0]], dtype=np.float32)
    bordas = cv2.filter2D(blur, ddepth=cv2.CV_32F, kernel=kernel)
    result = blur.astype(np.float32) - bordas
    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)
    if name:
        cv2.imwrite(os.getcwd()+"/result/"+str(var)+name, result)
    return result

def ideal_high_pass_vs_gaussian(img: np.ndarray, name: str = None) -> np.ndarray:
    freq_img = np.fft.fftshift(np.fft.fft2(img))
    row, columns = img.shape
    h = np.zeros((row,columns), dtype=np.float32)
    gp = np.zeros((row,columns), dtype=np.float32)
    gd0 = 80
    for u in range(row):
        for v in range(columns):
            val = np.sqrt((u-row/2)**2+(v-columns/2)**2)
            gp[u,v] = np.exp(-val**2/(2*gd0*gd0))
            if val <= 200:
                h[u,v] = 1
            else:
                h[u,v] = 0
    
    h = 1 - h
    gp = 1 - gp
    result = freq_img * h
    result_img = np.abs(np.fft.ifft2(np.fft.ifftshift(result)))
    result_img = img.astype(np.float32) - result_img.astype(np.float32)
    result_img = np.clip(result_img, 0, 255)
    result_img = result_img.astype(np.uint8)

    result_g = freq_img * gp
    result_g_img = np.abs(np.fft.ifft2(np.fft.ifftshift(result_g)))
    result_g_img = img.astype(np.float32) - result_g_img.astype(np.float32)
    result_g_img = np.clip(result_g_img, 0, 255)
    result_g_img = result_g_img.astype(np.uint8)

    blur = cv2.GaussianBlur(result_img, sigmaX=0.8, ksize=(3,3))
    result_img_combo = img.astype(np.float32) + result_img.astype(np.float32) - blur.astype(np.float32)
    result_img_combo = np.clip(result_img_combo, 0, 255)
    result_img_combo = result_img_combo.astype(np.uint8)
    
    if name:
        cv2.imwrite(os.getcwd()+"/result/"+name+"ideal.png", result_img)
        cv2.imwrite(os.getcwd()+"/result/"+name+"gauss.png", result_g_img)
        cv2.imwrite(os.getcwd()+"/result/"+name+"ideal+gaussBlur.png", result_img_combo)
    return result_img

dataList = inputData()
img = cv2.imread(dataList[0], dataList[1])
if img is not None:
    color_flag = dataList[1]
    choice = process_choice(color=color_flag)
    match choice:
        case 1:
            half_rows(img, color_flag)
        case 2:
            half_columns(img,color_flag)
        case 3:
            quarter_size(img, color_flag)
        case 4:
            double_rows(img, name="dobra_linhas.png")
        case 5:
            double_cols(img, name= "dobra_colunas.png")
        case 6:
            double_img(img, name="dobra_imagem.png")
        case 7:
            if color_flag:
                half_channels(img, name="downscale_img.png")
            else:
                call_laplacian(img,kernel_size=3,name="aplica_filtro_laplace.png")
        case 8:
            if color_flag:
                call_laplacian(img,kernel_size=3,name="aplica_filtro_laplace.png")
            else:
                var = float(input("Insira a variancia desejada: "))
                call_gaus_lap(img, kernel_size=3, var=var, name="_aplica_gaus_laplace.png")
        case 9:
            if color_flag:
                var = float(input("Insira a variancia desejada: "))
                call_gaus_lap(img, kernel_size=3, var=var, name="_aplica_gaus_laplace.png")
            else:
                ideal_high_pass_vs_gaussian(img, name="compara_ideal_")
        case 10:
            if color_flag:
                ideal_high_pass_vs_gaussian(img, name="compara_ideal_")
            else:
                print("Operacao Nao Reconhecida")
        case _:
            print("Operacao nao reconhecida.\n")
else:
    print("Erro: imagem não encontrada.")
