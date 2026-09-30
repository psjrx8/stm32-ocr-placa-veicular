import argparse
from array import array
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import shutil
from array import array


BASE = Path(__file__).resolve().parent

# INICIO FILTROS
# Um pixel de 8 bits só comporta 0..255. O realce pode sair dessa faixa;
# valores negativos viram preto e valores acima de 255 viram branco.
def saturar_u8(value):
    return max(0, min(255, value))


# A diferença tira a parte suave da imagem e deixa as mudanças locais.
# Em uma região uniforme, pixel e média são iguais e a resposta é zero.
def passa_alta(pixel, media):
    return int(pixel) - int(media)


# Somamos o detalhe de volta ao pixel original. Com ganho 1, fica
# pixel + (pixel - média), ou seja, 2*pixel - média. Depois limitamos a 0..255.
def combinar_realce(pixel, high, ganho_num=1):
    return saturar_u8(int(pixel) + high*ganho_num)


def realcar_inplace(gray, w, h, ganho_num=1, observar=None):
    validar(gray,w,h)
    # O parâmetro controlado aqui é só o ganho inteiro, entre 0 e 8.
    if not 0 <= ganho_num <= 8:
        raise ValueError('Ganho: numerador 0..8; denominador 1..16.')
    # Cada média 3x3 precisa da linha de cima, da atual e da seguinte.
    # Guardamos apenas essas três linhas: 3*160 = 480 bytes de pixels.
    rows = [bytearray(w) for _ in range(3)]
    # Carrega as primeiras linhas antes de modificar qualquer pixel de gray.
    for y in range(min(3,h)):
        rows[y][:] = gray[y*w:(y+1)*w]
    for y in range(h):
        # O resto por 3 escolhe o espaço do anel: 0, 1, 2, 0, 1, 2...
        # A linha atual aqui ainda contém os valores originais.
        current = rows[y%3]
        for x in range(w):
            pixel = current[x]
            # Na borda não existem os oito vizinhos. Mantendo smooth=pixel,
            # o passa-alta fica zero e o realce não altera esse ponto.
            smooth = pixel
            if 0 < y < h-1 and 0 < x < w-1:
                previous, following = rows[(y-1)%3], rows[(y+1)%3]
                # Soma a vizinhança 3x3: três pixels de cada uma das três linhas.
                total = (previous[x-1]+previous[x]+previous[x+1]
                         +current[x-1]+current[x]+current[x+1]
                         +following[x-1]+following[x]+following[x+1])
                # Divide pelos nove pixels. // descarta a fração;
                # a média usa somente os pixels originais guardados no anel.
                smooth = total//9
            high = passa_alta(pixel,smooth)
            value = combinar_realce(pixel,high,ganho_num)
            # O buffer é uma sequência de linhas. y*w pula as linhas anteriores
            # e +x encontra a coluna. Agora podemos substituir o pixel em gray.
            i=y*w+x
            gray[i]=value
            if observar is not None:
                observar(i,smooth,high,value)
        # Terminada a linha y, a próxima conta vai precisar da linha y+2.
        # Ela ainda não foi realçada. Seu conteúdo substitui a linha mais antiga.
        # Ex.: depois de y=1, descartamos a linha 0 e carregamos a linha 3.
        if y+2 < h:
            rows[(y+2)%3][:] = gray[(y+2)*w:(y+3)*w]

# FIM FILTROS

# INICIO PIPELINE
def filtrar(gray,w,h,ganho_num=1,observar=None):
    realcar_inplace(gray,w,h,ganho_num,observar)


# Aplica o mesmo tamanho e alinhamento a cada recorte, na ordem das caixas.
def normalizar_caracteres(binary,w,boxes):
    return [normalizar(binary,w,box) for box in boxes]


# Compara cada amostra com os templates permitidos naquela posição
# e junta os sete rótulos em uma única string.
def classificar_caracteres(samples,boxes,templates):
    details=[]
    for pos,(sample,box) in enumerate(zip(samples,boxes)):
        result=classificar(sample,templates,pos)
        result['box']=list(box)
        details.append(result)
    return ''.join(r['caractere'] for r in details)

# FIM PIPELINE

# INICIO IMPLEMENTACAO STM
# Cada caractere é colocado num quadro de 16x24: 384 valores binários.
# A string abaixo põe primeiro as 26 letras e depois os dez números.
NW, NH = 16, 24
ALFABETO = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


# A entrada é R,G,B,R,G,B...; a saída tem um único byte por pixel.
# Para 160x120, passamos de 57.600 bytes RGB para 19.200 bytes de cinza.
def grayscale(rgb):
    if len(rgb) % 3 != 0: # 3 entradas por pixel
        raise ValueError('RGB deve conter três bytes por pixel.')
    quantidade_pixels = len(rgb) // 3
    gray = bytearray(quantidade_pixels)
    for pixel in range(quantidade_pixels):
        # Cada pixel ocupa três posições consecutivas no buffer de entrada.
        i = pixel * 3
        r = rgb[i]
        g = rgb[i + 1]
        b = rgb[i + 2]
        # Y = 0,299*RED + 0,587*GREEN + 0,114*BLUE -> 77/256 = 0,301 | 150/256 = 0,586 | 29/256 = 0,113
        # Os pesos aproximam a conversão usual sem usar ponto flutuante.
        # Eles somam 256, então preto continua 0 e branco continua 255.
        # +128 arredonda antes da divisão. Para este valor não negativo,
        # >>8 equivale a dividir por 256 e descartar a parte fracionária.
        # Ex.: um pixel só vermelho (255,0,0) resulta em cinza 77.
        gray[pixel] = (77*r + 150*g + 29*b + 128) >> 8 # desloca 8 bits e trunca
    return gray


# Otsu escolhe o limiar a partir da imagem. Escuro vira alto=1; claro vira
# baixo=0. Cria um novo buffer e mantém o grayscale recebido intacto.
def binarizar(gray):
    t = otsu(gray)
    return bytearray(1 if v <= t else 0 for v in gray), t


# Testa onde separar o histograma em dois grupos, escuro e claro.
# Não reconhece letras: procura uma divisão com médias bem separadas.
# As médias são aproximadas em Q8 para manter a conta inteira.
def otsu(gray):
    # hist[50], por exemplo, conta quantos pixels têm intensidade 50.
    hist = [0]*256
    for v in gray:
        hist[v] += 1
    total = len(gray)
    # Intensidade vezes quantidade, somada para todos os níveis, equivale
    # à soma dos valores de todos os pixels da imagem.
    soma = sum(i*n for i, n in enumerate(hist))
    # n0: quantidade no grupo escuro; s0: soma das intensidades desse grupo.
    n0 = s0 = 0
    melhor_score, limiar = -1, 0
    # Para cada limiar t, o grupo escuro inclui 0..t e o claro inclui t+1..255.
    for t in range(256):
        n0 += hist[t]
        s0 += t*hist[t]
        n1 = total-n0
        # Ignora grupos vazios; não existe média quando a quantidade é zero.
        if not n0 or not n1:
            continue
        # Médias Q8, divisão positiva, determinístico em Python e C.
        # Multiplicar por 256 antes de dividir guarda a média em Q8.
        # Uma média 12,5 é representada por 3200, sem precisar de float.
        # s0 pertence ao grupo escuro; soma-s0 pertence ao grupo claro.
        m0 = (s0*256)//n0
        m1 = ((soma-s0)*256)//n1
        delta = m0-m1
        # A diferença ao quadrado mede a separação entre as médias.
        # n0*n1 pondera pelo tamanho dos grupos: diferença grande sozinha
        # não basta. O maior score escolhe o limiar; não é uma probabilidade.
        # A divisão inteira das médias aproxima o critério de Otsu.
        score = n0*n1*delta*delta
        if score > melhor_score:
            melhor_score, limiar = score, t
    return limiar


# Confere dimensões e quantidade de pixels
def validar(buf, w, h):
    if w <= 0 or h <= 0 or len(buf) != w*h or w*h > 65535:
        raise ValueError('Dimensões/buffer inválidos; máximo 65535 pixels.')


# Procura pontos escuros conectados, usando os oito vizinhos (diagonais
# incluídas). Ainda sem reconhecimento
def componentes(binary, w, h):
    validar(binary, w, h)
    work = bytearray(binary)
    queue = array('H', [0])*(w*h)
    boxes = []
    for start in range(w*h):
        # Fundo e pixels já visitados são zero. Um 1 inicia um novo ponto.
        if not work[start]:
            continue
        work[start] = 0
        queue[0] = start
        # head aponta o próximo item a ler; tail aponta o próximo espaço livre.
        # O pixel inicial já está na fila e marcado, para não entrar duas vezes.
        head, tail = 0, 1
        # %w recupera a coluna; //w recupera a linha. A caixa começa nesse pixel
        # e vai aumentando conforme encontramos o restante da mancha.
        x0 = x1 = start % w
        y0 = y1 = start // w
        while head < tail:
            i = queue[head]
            head += 1
            x, y = i % w, i // w
            x0, x1 = min(x0,x), max(x1,x)
            y0, y1 = min(y0,y), max(y1,y)
            # Visita a vizinhança sem sair da imagem. range não inclui o limite
            # final, por isso y+2 permite incluir y+1; o mesmo vale para x.
            for yy in range(max(0,y-1), min(h,y+2)):
                for xx in range(max(0,x-1), min(w,x+2)):
                    j = yy*w+xx
                    # Marca ao entrar na fila, não ao sair. Isso evita repetir
                    # o mesmo pixel quando ele é vizinho de vários outros.
                    if work[j]:
                        work[j] = 0
                        queue[tail] = j
                        tail += 1
        # Direita e baixo são exclusivos: +1 inclui o último pixel no recorte.
        # tail também é a área em pixels de tinta, porque cada um entrou uma vez.
        boxes.append((x0,y0,x1+1,y1+1,tail))
    return boxes


# Escolhe quais manchas têm tamanho e alinhamento compatíveis com a linha
# de sete caracteres deste projeto. Os limites abaixo são regras do ensaio.
def segmentar(binary, w, h, min_height=10):
    candidates = []
    for x0,y0,x1,y1,area in componentes(binary,w,h):
        cw, ch = x1-x0, y1-y0
        # Descarta qualquer ponto que encoste na borda, não só a moldura.
        # Por isso a ROI precisa deixar margem ao redor dos caracteres.
        if (x0 == 0 or y0 == 0 or x1 == w or y1 == h):
            continue  # Moldura conectada à borda da ROI.
        # Aceita altura >=10 por padrão e no máximo 80% do quadro.
        # Exige largura entre 1 e a altura e pelo menos oito pixels de tinta.
        # Essas regras rejeitam manchas pequenas e formas muito largas.
        if ch >= min_height and ch <= h*4//5 and 1 <= cw <= ch and area >= 8:
            candidates.append((x0,y0,x1,y1))
    # Ordena pelo lado esquerdo: a leitura da placa é da esquerda para a direita.
    candidates.sort(key=lambda b:b[0])
    if len(candidates) != 7:
        raise ValueError(f'Segmentação encontrou {len(candidates)} caracteres, esperados 7. '
                         'Ajuste ROI/limiar/min-height.')
    # Com sete alturas ordenadas, heights[3] é a mediana. Os centers abaixo
    # são o DOBRO do centro vertical: y0+y1, para evitar divisão por dois.
    heights = sorted(b[3]-b[1] for b in candidates)
    centers = [b[1]+b[3] for b in candidates]
    # Rejeita se a maior altura exceder o dobro da menor, ou se a diferença
    # entre os centros verticais exceder metade da altura mediana.
    # A segunda comparação usa centros dobrados, por isso não divide aqui.
    if min(heights)*2 < max(heights) or max(centers)-min(centers) > heights[3]:
        raise ValueError('Componentes não formam uma linha de caracteres consistente.')
    return candidates


# Faz o recorte caber em 16x24 preservando sua proporção e centralizando.
# Assim, uma letra estreita não fica artificialmente larga como as demais.
def normalizar(binary, w, box):
    x0, y0, x1, y1 = box
    largura = x1 - x0
    altura = y1 - y0
    if largura <= 0 or altura <= 0:
        raise ValueError('Recorte vazio.')
    # Define o tamanho que cabe no quadro sem esticar.
    if largura * NH >= altura * NW:
        nova_largura = NW
        nova_altura = max(1, altura * NW // largura)
    else:
        nova_altura = NH
        nova_largura = max(1, largura * NH // altura)

    # Centraliza no quadro de saída.
    inicio_x = (NW - nova_largura) // 2
    inicio_y = (NH - nova_altura) // 2
    
    # Começa preenchido com zero (fundo). Sobras ficam como margem ao redor.
    normalizada = bytearray(NW * NH)
    for y in range(nova_altura):
        # Para cada ponto da saída, busca um ponto do recorte original.
        # É vizinho mais próximo com índices inteiros: não mistura pixels.
        origem_y = y0 + y * altura // nova_altura
        for x in range(nova_largura):
            origem_x = x0 + x * largura // nova_largura
            origem = origem_y * w + origem_x
            # Aplica o deslocamento que centraliza o caractere no quadro.
            destino = (inicio_y + y) * NW + inicio_x + x
            normalizada[destino] = binary[origem]
    return normalizada


# Conta posições diferentes: igualdade soma 0, diferença soma 1.
# É distância de Hamming. Menos diferenças significa um encaixe melhor.
def comparar(sample, template):
    return sum(a != b for a,b in zip(sample, template))


# Padrão adotado: LLLNLNN. pos começa em zero, então a quinta posição é 4.
# A posição restringe candidatos, mas não garante que a letra escolhida esteja certa.
def classificar(sample, templates, pos=0):
    allowed = ALFABETO
    letra = pos < 3 or pos == 4 # letra nas posicoes 0,1,2 e 4
    allowed = ALFABETO[:26] if letra else ALFABETO[26:] # primeiras 26 letras ou últimos 10 números

    # Calcula a distância de cada template permitido e ordena da menor para
    # a maior. Em empate, desempata pelo rótulo em ordem de caracteres.
    ranking = sorted((comparar(sample,t),c) for c,t in templates.items() if c in allowed)
    if not ranking:
        raise ValueError('Nenhum template para esta posição.')
    score,char = ranking[0]
    # margem = distância do segundo - distância do primeiro. Uma margem maior
    # indica separação maior entre candidatos, não uma chance de acerto.
    second = ranking[1][0] if len(ranking)>1 else score
    return dict(caractere=char, erros=score, pixels=NW*NH, margem=second-score)


## FIM IMPLEMENTACAO STM
# Aplica grayscale, realce, binarização e normalização, como nas amostras.
# Os arquivos são identificados pelo nome: A.png, B.png, ..., 0.png, ..., 9.png.
def carregar_templates(ganho_num=1):
    templates = {}
    for c in ALFABETO:
        path = BASE/'templates'/ (c +'.png')
        if not path.exists():
            continue
        im = Image.open(path).convert('RGB')
        w,h = im.size
        gray = grayscale(im.tobytes())
        filtrar(gray,w,h,ganho_num)
        b,_ = binarizar(gray)
        # Encontra todos os pixels de tinta. Seus extremos definem o recorte justo
        points = [(i%w,i//w) for i,v in enumerate(b) if v]
        if not points:
            raise ValueError(f'Template vazio: {path}')
        box = (min(x for x,y in points),min(y for x,y in points),
               max(x for x,y in points)+1,max(y for x,y in points)+1)
        templates[c] = normalizar(b,w,box)
    if not templates:
        raise ValueError('Pasta de templates vazia.')
    return templates


# No algoritmo, tinta é 1 e fundo é 0. Invertemos no png:
# tinta vira preto (0) e fundo vira branco (255).
def salvar_binaria(binary,w,h,path):
    Image.frombytes('L',(w,h),bytes(0 if v else 255 for v in binary)).save(path)


# L indica um canal de cinza de 8 bits. w e h organizam o buffer em linhas.
# A gravação é diagnóstico do PC, sem alterar os valores usados pelo algoritmo.
def salvar_gray(buf,w,h,path):
    Image.frombytes('L',(w,h),bytes(buf)).save(path)


# A resposta H varia de -255 a 255, mas o PNG precisa de valores 0..255.
# (H+256)//2 traz zero para 128; negativo fica escuro e positivo fica claro.
def salvar_passa_alta(high,w,h,out):
    # Visualização: zero=128; sinal preservado, escala FIXA de 1/2.
    display=bytearray((int(v)+256)//2 for v in high)
    salvar_gray(display,w,h,out/'08_passa_alta_visual.png')
    """ # raw do passa_alta

    import struct

    with (out/'passa_alta_i16le.raw').open('wb') as f:

        for value in high:

            f.write(struct.pack('<h',value))

    """


# Corrige orientação de fotos, garante RGB e prepara dois enquadramentos.
# A ROI usa coordenadas da ORIGINAL; pad preserva proporção e acrescenta branco.
# NEAREST escolhe pixels sem interpolar médias.
def adquirir_imagem(path,roi):
    with Image.open(path) as src:
        original=ImageOps.exif_transpose(src).convert('RGB')
    selected=original
    image=ImageOps.pad(selected,(160,120),method=Image.Resampling.NEAREST,color='white')
    # Sem ROI, as duas variáveis apontam para a mesma imagem.
    image_roi=image
    if roi:
        if roi[2]>original.width or roi[3]>original.height:
            raise ValueError('ROI fora da imagem original.')
        selected=original.crop(roi)
        image_roi = ImageOps.pad(selected,(160,120),method=Image.Resampling.NEAREST,color='white')
    return original,image,image_roi


# Executa cada etapa e salva a evolução da imagem para comparar os resultados.
def run(args):
    out= BASE/'resultados'/args.imagem.stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Crop da imagem
    ROI_PADRAO = (80, 80, 640, 200) # x_esquerda, y_superior, x_direita, y_inferior
    original, image, image_crop = adquirir_imagem(args.imagem, roi=ROI_PADRAO)
    original.save(out / '01_original.png')
    image.save(out / '02_entrada_160x120.png')
    image_crop.save(out / '03_entrada_crop_160x120.png')

    # Escala de cinza
    gray = grayscale(image_crop.tobytes())
    salvar_gray(gray,160,120,out/'04_grayscale.png')
    # (out / 'entrada_grayscale.raw').write_bytes(gray) #Bytes para o stm
    
    # Visualizar suavização e passa alta
    # Estes frames completos servem para diagnóstico no PC. O filtro em si
    # usa o anel de três linhas. A resposta passa-alta precisa aceitar negativos.
    # Anterior, atual, posterior
    suavizada = bytearray(160 * 120)
    passa_alta = array('h', [0]) * (160 * 120)
    def registrar_etapas(indice, valor_suavizado, valor_passa_alta, valor_realcado):
        suavizada[indice] = valor_suavizado
        passa_alta[indice] = valor_passa_alta

    # Referência visual antes do realce. Esta binária é salva, mas não é
    # segmentada/classificada neste run; o reconhecimento usa a versão realçada.
    binaria_sem_realce, t_limiar_sem_realce= binarizar(gray)
    salvar_binaria(binaria_sem_realce,160,120,out/'05_binaria_sem_realce.png')

    GANHO_PADRAO = 1
    # A partir daqui, gray contém o realce. O ganho 1 soma uma vez o passa-alta.
    filtrar(gray,160,120,GANHO_PADRAO,registrar_etapas)
    salvar_gray(suavizada,160,120,out/'06_suavizada.png')
    salvar_gray(gray,160,120,out/'07_grayscale_realcada.png')
    salvar_passa_alta(passa_alta,160,120,out)
    # (out/'realcada_u8.raw').write_bytes(gray)

    # Otsu calcula outro limiar para a nova imagem. A comparação dos PNGs
    # inclui o efeito do realce e da escolha automática de limiar.
    binaria_com_realce, t_limiar_com_realce = binarizar(gray)
    salvar_binaria(binaria_com_realce,160,120,out/'09_binaria_com_realce.png')
    # (out/'binaria_01.raw').write_bytes(binary)

    # Selecionar os componentes considerados caracteres.
    ALTURA_MINIMA = 10
    boxes = segmentar(binaria_com_realce,160,120, ALTURA_MINIMA)
    # Desenha sobre uma cópia do mesmo enquadramento usado no cálculo.
    # Essas caixas são diagnóstico; não fazem parte da binária.
    segment = image_crop.copy()
    draw=ImageDraw.Draw(segment)
    for i,(x0,y0,x1,y1) in enumerate(boxes):
        # O retângulo inclui o ponto final, enquanto a caixa de recorte não.
        # Por isso subtraímos 1 à direita e embaixo.
        draw.rectangle((x0,y0,x1-1,y1-1),outline='red')
        draw.text((x0,max(0,y0-12)),str(i+1),fill='red')
    segment.save(out/'10_segmentada.png')
    amostras = normalizar_caracteres(binaria_com_realce,160,boxes)
    for i,amostra in enumerate(amostras):
        salvar_binaria(amostra,NW,NH,out/f'12_char_{i+1}.png')
        # (out/f'char_{i+1}_01.raw').write_bytes(amostra)

    # Usa o mesmo ganho nos templates. A altura dos caracteres de referência
    # também é compatível com a das amostras, para o filtro agir de modo parecido.
    templates = carregar_templates(GANHO_PADRAO)
    placa = classificar_caracteres(amostras,boxes,templates)
    print('Placa:', placa)
    print('Resultados:',out.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('imagem',type=Path)
    args = p.parse_args()
    try:
        run(args)
    except (ValueError,OSError) as e:
        p.exit(2,f'Erro: {e}\n')


if __name__ == '__main__':
    main()
