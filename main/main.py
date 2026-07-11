import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

start = Path.cwd().parent
img_path = start / "images"
img_output_path = img_path / "output"

"""
Aplica um ciclo completo de compressao e descompressao JPEG na imagem de entrada.
INPUT:
    img - imagem a ser aplicada o ciclo
    quality - valor de qualidade a ser aplicada pelo JPEG
OUTPUT:
    None - Nao foi possivel aplicar o ciclo na imagem
    OU
    decoded - imagem apos aplicacao do ciclo com o nivel de qualidade exigida
"""
def jpeg_cycle(img: np.ndarray, quality: int) -> np.ndarray:
    rval, rmem = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if rval:
        decoded = cv2.imdecode(rmem, 1)
        return decoded
    else:
        raise RuntimeError("Incapaz de executar ciclo de compressao e descompressao JPEG.")

"""
Aplica n ciclos JPEG em uma imagem ate atingir o ponto fixo ou chegar ao maximo de iteracoes.
INPUT:
    img - imagem a ser transformada em tamper-evident
    quality - valor de qualidade exigida para o ciclo JPEG
    max_iters - valor maximo de iteracoes a serem executadas
    tol - valor maximo de tolerancia de diferenca entre pixels
OUTPUT:
    None - imagem nao convergiu ao ponto fixo no valor maximo de iteracoes
    OU
    current_img - imagem em seu ponto fixo apos i iteracoes
"""
def create_tamper_evident(img: np.ndarray, quality: int, max_iters: int = 200, tol: int = 0) -> np.ndarray:
    current_img = img.copy()
    for i in range(max_iters):
        next_img = jpeg_cycle(current_img, quality)
        img_dif = np.abs(next_img.astype(np.uint8) - current_img.astype(np.uint8))
        num_dif = np.any(img_dif > tol, axis=-1).sum()
        print(f"Iteracao atual: {i}, Pixeis alterados: {num_dif}")
        if(num_dif == 0):
            print(f'Ponto fixo atingido em i={i}.')
            return current_img
        current_img = next_img
    raise RuntimeWarning("Imagem nao convergiu dentro do numero de iteracoes")
    return current_img

"""
Calcula a variacao entre os blocos JPEG de uma imagem apos a aplicacao de um ciclo JPEG
Sempre considera blocos 8x8 para verificacao
INPUT:
    img - imagem a ser testada
    quality - valor de qualidade exigida para o ciclo JPEG
OUTPUT:
    mask_array - array de blocos alterados na imagem
    diff_array - array de diferenca pixel a pixel entre imagem alterada e original
"""
def check_tamper(img: np.ndarray, quality: int):
    h,w,_ = img.shape
    # reducao da imagem caso nao seja multiplo de 8
    h-=h%8
    w-=w%8
    img_arr = img[:h, :w].astype(np.uint8)
    img_tamper = jpeg_cycle(img_arr, quality).astype(np.uint8)

    diff_array = np.any(np.abs(img_arr-img_tamper) > 0, axis=-1)
    mask_array = np.zeros((h//8, w//8), dtype=bool)

    for block_y in range(mask_array.shape[0]):
        for block_x in range(mask_array.shape[1]):
            block = diff_array[block_y*8:(block_y+1)*8,
                               block_x*8:(block_x+1)*8]
            mask_array[block_y, block_x] = block.any()
    return mask_array, diff_array

"""
"""
def show_diff(img: np.ndarray, mask: np.ndarray) -> None:
    if mask.sum() != 0:
        h = mask.shape[0]*8
        w = mask.shape[1]*8

        img_crop = img[:h,:w].copy()
        red = np.zeros((h,w),dtype=bool)

        for block_y in range(mask.shape[0]):
            for block_x in range(mask.shape[1]):
                if(mask[block_y, block_x]):
                    red[block_y*8:(block_y+1)*8,
                        block_x*8:(block_x+1)*8] = True
     
        red_layer = np.zeros_like(img_crop)
        red_layer[:, :] = [0,0,255]
        
        blended = cv2.addWeighted(img_crop, 1-0.5, red_layer, 0.5, 0)
        result = img_crop.copy()
        result[red] = blended[red]

        print(f'Quantidade de blocos alterados: {mask.sum()/mask.size}')
        cv2.imwrite(img_output_path/"changed.jpg", result)
    else:
        print("Nao foi detectada nenhuma alteracao na imagem")

quality = 90
img = cv2.imread(img_path/"SIPI"/"4.1.03.BMP")
te_img = create_tamper_evident(img=img, quality=quality)
cv2.imwrite(img_output_path/"tamper_evident.jpg",te_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

sus_img = cv2.imread(img_output_path/"tamper_evident.jpg")
mask, diff = check_tamper(img=sus_img, quality=quality)
print(f"Quantidade de blocos alterados: {mask.sum()} de {mask.size}")
show_diff(img=sus_img, mask=mask)

"""
test_img = cv2.imread(img_path/"face.png",1)
diff_array = []
if test_img is not None:
    cv2.imwrite(img_output_path/"image.jpg",test_img,[cv2.IMWRITE_JPEG_QUALITY, 10])
    new_img = cv2.imread(img_output_path/"image.jpg",1)
    dif = (test_img != new_img).sum()
    print(dif)
    diff_array.append(dif)
    while(dif != 0 ):
        cv2.imwrite(img_output_path/"image.jpg",test_img,[cv2.IMWRITE_JPEG_QUALITY, 10])
        prev_img = new_img
        new_img = cv2.imread(img_output_path/"image.jpg",1)
        dif = (prev_img != new_img).sum()
        diff_array.append(dif)
        print(dif)
"""