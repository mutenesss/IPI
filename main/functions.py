import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

start = Path.cwd().parent
img_path = start / "images"
img_output_path = img_path / "output"
convergence_index = []
convergence_list = []
"""
subdirs = [x for x in img_path.iterdir() if x.is_dir() and x != img_output_path]
fileList = [x for x in img_path.iterdir() if x.is_file()]
for x in fileList:
    print(x.name)
print(subdirs)
"""
"""
Retira pixeis da imagem para manter a mesma com tamanho multiplo de 8
INPUT:
    img - imagem a sofrer o corte de tamanho
OUTPUT:
    result - imagem apos retirada de pixeis
"""
def resize_img(img: np.ndarray) -> np.ndarray:
    h,w = img.shape[:2]
    h8 = h - (h%8)
    w8 = w - (w%8)
    result = img[:h8, :w8]
    return result

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
        raise RuntimeError("Incapaz de executar ciclo de compressao e descompressao JPEG.\n")

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
def create_tamper_evident(img: np.ndarray, quality: int, max_iters: int = 100, tol: int = 0) -> (np.ndarray | None):
    current_img = img.copy()
    index_list = []
    for i in range(max_iters):
        next_img = jpeg_cycle(current_img, quality)
        img_dif = np.abs(next_img.astype(np.uint16) - current_img.astype(np.uint16))
        num_dif = np.any(img_dif > tol, axis=-1).sum()
        del img_dif
        print(f"Iteracao atual: {i}, Pixeis alterados: {num_dif}")
        index_list.append(int(num_dif))
        if(num_dif == 0):
            print(f'Ponto fixo atingido em i={i}.')
            convergence_index.append(i)
            convergence_list.append(index_list)
            return current_img
        current_img = next_img
    print("Imagem nao convergiu dentro do numero de iteracoes\n")
    return None

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
    del img_arr
    del img_tamper
    mask_array = np.zeros((h//8, w//8), dtype=bool)
    mask_array = diff_array.reshape(h//8, 8, w//8, 8). any(axis=(1,3))
    return mask_array, diff_array

"""
Gera a imagem com alteracoes caso existam
INPUT:
    img - imagem considerada alterada
    mask - mascara de alteracoes computada por check_tamper
OUTPUT:
"""
def show_diff(img: np.ndarray, mask: np.ndarray, outputPath: Path) -> None:
    if mask.sum() != 0:
        h = mask.shape[0]*8
        w = mask.shape[1]*8

        img_crop = img[:h,:w].copy()
        red = np.zeros((h,w),dtype=bool)
        del h
        del w

        for block_y in range(mask.shape[0]):
            for block_x in range(mask.shape[1]):
                if(mask[block_y, block_x]):
                    red[block_y*8:(block_y+1)*8,
                        block_x*8:(block_x+1)*8] = True
     
        red_layer = np.zeros_like(img_crop)
        red_layer[:, :] = [0,0,255]
        
        blended = cv2.addWeighted(img_crop, 0.5, red_layer, 0.5, 0)
        result = img_crop.copy()
        del img_crop
        result[red] = blended[red]

        print(f'Porcentagem de blocos alterados: {mask.sum()/mask.size}')
        fileName = (img_output_path/outputPath.stem)
        fileName = fileName.with_name(fileName.stem + "_changed.jpg")
        cv2.imwrite(fileName, result)
        del result
        print("Detectada alteracao na imagem.\n")
    else:
        print("Nao foi detectada nenhuma alteracao na imagem.\n")


if __name__ == "__main__":
    subdirs = [x for x in img_path.iterdir() if x.is_dir() and x != img_output_path]
    subdirs.append(img_path)
    quality = 90
    fileList = [x for x in img_path.iterdir() if x.is_file()]
    failed = []
    fileOrder = []
    for dirs in subdirs:
        fileList = [x for x in dirs.iterdir() if x.is_file]
        for file in fileList:
            if file.suffix:
                print(f"Testando arquivo: {file.name}")
                print(f"Diretorio: {dirs}\n")
                fileOrder.append(str(file.name))
                cur_img = cv2.imread(file)

                crop_img = resize_img(img=cur_img)
                del cur_img
                tamper_img = create_tamper_evident(img=crop_img, quality=quality)
                if tamper_img is not None:
                    del crop_img
                    tamper_img = resize_img(img=tamper_img)

                    output_name = (img_output_path/file.stem)
                    output_name = output_name.with_name(output_name.stem + "_tamper.jpg")
                    cv2.imwrite(output_name, tamper_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    del tamper_img

                    test_img = cv2.imread(output_name)
                    change_img = cv2.blur(test_img, (500,500))
                    mask, diff = check_tamper(img=change_img, quality=quality)
                    del change_img
                    #mask, diff = check_tamper(img=test_img, quality=quality)
                    print(f"Quantidade de blocos alterados: {mask.sum()} de {mask.size}\n")
                    show_diff(img=test_img, mask=mask, outputPath=output_name)
                    del test_img
                else:
                    failed.append(file.name)

    print("Os seguintes arquivos falharam em convergir:\n")
    for fail in failed:
        print(fail)

    print("Criando arquivo de indices de convergencia.\n")
    with open("convergence_index.txt", "w") as f:
        f.write(",\n".join(map(str,convergence_index)))

    print("Criando arquivo de listas de convergencia.\n")
    #np.savetxt("convergence_list.txt", convergence_list, fmt='%d', delimiter=',')
    with open("convergence_list.txt", "w") as f:
        f.write(",\n".join(map(str,convergence_list)))
        
    for i in range(0,len(fileOrder)):
        print(f"{i}, {fileOrder[i]}")