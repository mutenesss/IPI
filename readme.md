# Trabalho 2
### Instalar bibliotecas necessárias utilizando:

```bash
pip install -r requirements.txt
```

O programa `main.py` executa a segmentação da imagem "onion.jpg" utilizando o algoritmo `k-means` e melhora a visualização do tumor presente na imagem "brain.jpg" utilizando binarização e thresholding global.

### Execute o programa utilizando o comando:

```bash
python main.py
```

## Não é necessário entrada de dados para execução do programa, apenas a presença das imagens "brain.jpg" e "onion.jpg" na pasta *images*

### O resultado do programa segue a estrutura abaixo
```bash
.
├── images
│   ├── brain.jpg
│   ├── brain.zip
│   ├── onion.jpg
│   └── output                          # Resultados das operações morfológicas e binarização
│       ├── hist                        # Histogramas gerados pelo código
│       ├── kmeans                      # Imagens resultantes e centroides finais para cada k
│       │   ├── pp_center               
│       │   └── random_center
│       ├── original_enhanced.png       # Imagem resultante da união da imagem binarizada e original
│       ├── pp_means.gif                # União de todos os resultados para centroides provaveis
│       └── rand_means.gif              # União de todos os resultados para centroides aleatorios
├── main.py
├── Projeto_2.pdf
├── readme.md
└── requirements.txt
```