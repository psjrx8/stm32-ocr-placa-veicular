# LAB 3 — reconhecimento de caracteres em placas

Programa em Python para reconhecer os sete caracteres de uma placa de teste, com fonte conhecida e enquadramento fixo. A ideia é validar o processamento no computador e usar os cálculos como referência para a futura implementação no STM32F411.

O código está em um único arquivo, `main.py`. Ele usa:

- **Python:** laços, buffers e cálculos do processamento.
- **Pillow:** abrir, recortar, redimensionar e salvar imagens; desenhar as caixas dos caracteres.
- **Biblioteca padrão:** `argparse` recebe o caminho da imagem, `pathlib` organiza caminhos, `array` guarda valores com sinal e índices, e `shutil` recria a pasta de saída.

Só o Pillow precisa ser instalado separadamente. Não há OCR pronto nem rede neural.

## Preparar e executar

Tenha Python 3 instalado. No Windows, abra um terminal na pasta do projeto e instale o Pillow:

```powershell
py -m pip install Pillow
```

Esse comando instala a biblioteca de imagens no Python chamado por `py`.

Coloque `main.py` na pasta do projeto, mantenha seus templates em `templates/` e suas imagens de teste em `imagens/`.

Em `templates/`, cada PNG deve conter um único caractere escuro em fundo claro, com o mesmo estilo das placas: `A.png`, `B.png`, ... `Z.png`, `0.png`, ... `9.png`. Use os recortes anteriores ao processamento; o carregador já aplica o filtro, binariza e normaliza. Arquivos ausentes são ignorados pelo código, então confira se todas as classes que você quer reconhecer estão presentes.

Execute informando a imagem:

```powershell
py main.py imagens/placa_01.png
```

A placa reconhecida aparece no terminal. Os PNGs de cada etapa ficam em `resultados/placa_01/`. Ao executar novamente com esse nome, a pasta de saída anterior é apagada e recriada. Isso também vale para imagens com o mesmo nome em diretórios diferentes.

## O que o programa faz

1. Abre a imagem, recorta a região dos caracteres e ajusta para 160×120, preservando a proporção e preenchendo as sobras com branco.
2. Converte RGB em cinza com pesos inteiros.
3. Salva uma binarização sem realce para comparação visual.
4. Calcula uma média 3×3, subtrai essa média do pixel original e soma a diferença de volta à imagem. Com ganho 1: `realçada = 2 × original − média`, limitada a 0–255.
5. Binariza a imagem realçada com limiar automático de Otsu.
6. Encontra componentes conectados, seleciona sete candidatos e ordena da esquerda para a direita.
7. Coloca cada caractere em um quadro de 16×24 sem esticá-lo e compara com os templates, contando pixels diferentes.

A classificação considera o padrão `LLLNLNN`: letras nas posições 1, 2, 3 e 5; números nas posições 4, 6 e 7 conforme placas automotivas do mercosul.

O código usa `ROI_PADRAO = (80, 80, 640, 200)`, medida na imagem original. Os modelos devem ter os caracteres completos nessa região. Para outro enquadramento, ajuste essa constante. O ganho é 1 e a altura mínima aceita na segmentação é 10 pixels.

Os resultados incluem original, enquadramentos, grayscale, suavização, passa-alta, imagem realçada, duas binárias, caixas de segmentação e sete caracteres individuais. Imagens criadas individualmente para fins de validação antes de embarcar o código
