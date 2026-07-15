# Trabalho 3
### Instalar bibliotecas necessárias utilizando:

```bash
pip install -r requirements.txt
```
### Artigo Base
**Tamper-evident Image using JPEG Fixed Points**

### Implementação

As funções que implementam a compressão de um conjunto de imagens e a geração de imagens _tamper-evident_ estão presentes no arquivo `functions.py`
Um exemplo de uso e teste utilizando a imagem "lena_color_512.tif" pode ser visto no arquivo `change_images.py`

O programa `main.py` executa a segmentação da imagem "onion.jpg" utilizando o algoritmo `k-means` e melhora a visualização do tumor presente na imagem "brain.jpg" utilizando binarização e thresholding global.

Os gráficos e curvas calculadas e apresentadas no relatório estão presentes na pasta *curvas*

### Execute o programa utilizando o comando:

```bash
python functions.py
```
ou
```bash
python change_images.py
```

## Não é necessário entrada de dados para execução do programa, apenas a presença das imagens a serem analisadas na pasta *images*

### O resultado do programa segue a estrutura abaixo
```bash
.
├── curvas
│   ├── convergencia_qualidades.png
│   ├── curva_comparativa.png
│   ├── curvas_convergencia.png
│   └── histograma_convergencia.png
├── images
│   ├── 5.3.01.BMP
│   ├── I03.BMP
│   ├── lena_color_512.tif
│   ├── output
│   ├── SIPI
│   └── tid2008
├── main
│   ├── change_images.py
│   ├── convergence_index.txt
│   ├── convergence_list.txt
│   ├── functions.py
│   ├── readme.md
│   └── requirements.txt
├── Relatorio_Trabalho_Final.pdf
├── Slide_Trabalho_Final.pdf
└── link_github.txt
```