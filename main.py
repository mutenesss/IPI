import numpy as np
import matplotlib.pyplot as plt
import os

# Ignore opencv warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import cv2

current = os.getcwd()
brain_path = current + "/images/brain.jpg"
onion_path = current + "/images/onion.jpg"
ret_path = current + "/images/output/"

"""
Leitura de imagens para o programa
Entrada:

Saída:
    Lista com imagens lidas
"""
def check_paths() -> list:
    path_list = []
    brain_img = cv2.imread(brain_path, 0)
    onion_img = cv2.imread(onion_path, 1)
    if brain_img is not None:
        path_list.append(brain_img)
    if onion_img is not None:
        path_list.append(onion_img)
    return path_list

"""
Passagem de filtro low_pass basico do OpenCV, em conjunto ao filtro de mediana
Entrada:
    img → Array com a imagem
    size → tamanho desejado do kernel, com size > 1 e impar
Saída:
    Imagem após filtragem low_pass e median
"""
def low_pass_median(img: np.ndarray, size: int) -> np.ndarray:
    kernel = np.ones((size,size), np.float32)/(size*size)
    result = cv2.filter2D(img, -1, kernel=kernel)
    median_result = cv2.medianBlur(result, size)
    cv2.imwrite(ret_path+'low_median.jpg',result)
    return median_result

"""
Criação de Histogramas para análise da imagem
Entrada:
    img → Array com a imagem
    name → Nome a ser adicionado ao caminho de retorno da string
Saída:
    Na pasta /images/output/hist/ são criadas 3 imagens que representam o histograma da imagem, sendo:
    name+_histNoLimit = Histograma sem limite no eixo Y
    name+_histYx = Histograma com o eixo Y limitado até o valor X, sendo X=1000 e X=10000
"""
def create_hist(img: np.ndarray, name: str) -> None:
    hist_ret = ret_path + "/hist/" + name
    histSize=256
    plt.hist(img.flatten(), bins=histSize, range=(0,histSize))
    plt.savefig(hist_ret + '_histNolimit.png')
    axes = plt.gca()
    axes.set_ylim([0,1000])
    plt.savefig(hist_ret + '_histY1000.png')
    axes.set_ylim([0,10000])
    plt.savefig(hist_ret + '_histY10000.png')

"""
Binarização da imagem de entrada utilizando um threshold global
Entrada:
    img → Array com a imagem
    threshold → Threshold para binarização da imagem
Saída:
    result1 → Imagem resultante da binarização feita com o threshold
    Na pasta /images/output/ é criada 1 imagem com o nome "threshold_bin_thresh.png" que é a mesma que result1 
"""
def binarize(img : np.ndarray, threshold: float) -> np.ndarray:
    retval, result1 = cv2.threshold(img, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
    cv2.imwrite(ret_path+str(threshold)+"_bin_thresh.png", result1)
    return result1

"""
Execução de operações morfológicas de abertura e fechamento em uma imagem binarizada
Entrada:
    img → Array com a imagem binarizada
    kernelSize → Tamanho do kernel desejado
Saída:
    opencloseImgRound → Resultado das operações de abertura e fechamento da imagem utilizando um kernel em elipse
    A mesma imagem resultante é salva em /images/output/ com o nome "kernelSize_open_closeRound.png"
"""
def morphOp(img: np.ndarray, kernelSize: int) -> np.ndarray:
    roundKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize,kernelSize))
    openImgRound = cv2.morphologyEx(img, cv2.MORPH_OPEN, roundKernel)
    opencloseImgRound = cv2.morphologyEx(openImgRound, cv2.MORPH_CLOSE, roundKernel)
    cv2.imwrite(ret_path+str(kernelSize)+'_open_closeRound.png', opencloseImgRound)
    return opencloseImgRound

"""
Faz a união dos componentes conexos resultantes das operações morfológicas com a imagem original
Entrada:
    orig → Imagem original monocromática sem quaisquer operação
    bin → Imagem binária resultante das operações morfológicas
Saída:
    Na pasta /images/output/ é salva uma imagem com o nome "original_enhanced.png", resultante da execução
    do algoritmo de componentes conexos e um OR bit a bit com a imagem original.
"""
def getConnected(orig: np.ndarray, bin: np.ndarray) -> None:
    result = np.zeros(bin.shape, dtype="uint8")
    compStats = cv2.connectedComponentsWithStats(bin,8,cv2.CV_32S)
    (labels, label_id, values, centroid) = compStats
    for i in range(1,labels):
        area = values[i, cv2.CC_STAT_AREA]
        if area > 100:
            mask = (label_id == i).astype("uint8")*255
            result = cv2.bitwise_or(orig, mask)

    cv2.imwrite(ret_path+"original_enhanced.png", result)

"""
Executa o algoritmo k-means em uma imagem colorida
A principal variação entre as duas execuções da função kmeans nesta função é o valor dos centroides utilizados na 
primeira execução, com:
    pp_centers →  calcula os valores prováveis dos centroides para a imagem 
    random_centers → calcula valores para centroides aleatórios para a imagem
Dessa forma, os valores calculados por pp_centers são facilmente repetíveis utilizando a mesma imagem, enquanto random_centers é aleatório
Entrada:
    img → Array da imagem
Saída:
    Nas pastas /images/output/kmeans/pp_center e /images/output/kmeans/random_center são salvas as imagens
    e os centroides calculados pela função kmeans, variando o K de 2 até 32
    As imagens são salvas com o nome "i_means.png" e o arquivo txt com os centroides é nomeado "i_centers.txt"
"""
def findKmeans(img: np.ndarray) -> None:
    reshaped_img = img.reshape((-1,3))
    reshaped_img = np.float32(reshaped_img)
    stop = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1)
    for i in range(2,33):
        ret, label, centers = cv2.kmeans(
            reshaped_img, i, None, stop, 2, cv2.KMEANS_PP_CENTERS
        )
        np.savetxt(ret_path+"/kmeans/pp_center/"+str(i)+"_centers.txt", centers)
        centers = np.uint8(centers)
        result = centers[label.flatten()]
        result = result.reshape((img.shape))
        cv2.imwrite(ret_path+"/kmeans/pp_center/"+str(i)+"_means.png",result)

        ret, label, centers = cv2.kmeans(
            reshaped_img, i, None, stop, 2, cv2.KMEANS_RANDOM_CENTERS
        )
        np.savetxt(ret_path+"/kmeans/random_center/"+str(i)+"_centers.txt", centers)
        centers = np.uint8(centers)
        result = centers[label.flatten()]
        result = result.reshape((img.shape))
        cv2.imwrite(ret_path+"/kmeans/random_center/"+str(i)+"_means.png",result)

    

img_list = check_paths()    
filter = low_pass_median(img_list[0],3)
create_hist(filter, "brain")
bin_img = binarize(filter,120)
morph = morphOp(bin_img,7)
getConnected(img_list[0], morph)
findKmeans(img_list[1])

"""
    Kmeans -> PP_Centers -> 20 means, sendo possivel argumentar separacao a partir de 16
    Kmeans -> Random_Centers -> 18 means, sendo possivel argumentar separacao a partir de 14
    
"""
