# ============================================================================
# IMPORTAÇÕES
# ============================================================================

# json: biblioteca para trabalhar com formato JSON (serialização/deserialização)
import json

# pipeline: função do Transformers (Hugging Face) que facilita o uso de modelos de ML
# Ela abstrai o processo de tokenização, inferência e pós-processamento
from transformers import pipeline

# FastMCP: framework para criar servidores MCP (Model Context Protocol)
# Permite expor ferramentas de IA que podem ser consumidas por clientes MCP
from fastmcp import FastMCP

# ============================================================================
# INICIALIZAÇÃO DO SERVIDOR MCP
# ============================================================================

# Cria uma instância do servidor MCP com o nome 'mcp-analise-sentimentos'
# Este nome identifica o servidor quando clientes MCP se conectam a ele
servidor_mcp = FastMCP('mcp-analise-sentimentos')

# ============================================================================
# VARIÁVEIS GLOBAIS
# ============================================================================

# classificador: armazena o modelo de classificação de emoções em memória
# Inicializado como None e carregado sob demanda para otimizar o tempo de inicialização
# Usar uma variável global evita recarregar o modelo (operação cara) a cada requisição
classificador = None

# ============================================================================
# DICIONÁRIO DE TRADUÇÃO
# ============================================================================

# Mapeia as 28 emoções do modelo GoEmotions (Google) do inglês para português
# O GoEmotions é um dataset com comentários do Reddit rotulados com emoções
# Este dicionário permite retornar resultados em português para melhor UX
TRADUCAO_EMOCOES = {
    'admiration': 'admiração',        # Sentimento de respeito e aprovação
    'amusement': 'diversão',          # Algo engraçado ou divertido
    'anger': 'raiva',                 # Irritação intensa
    'annoyance': 'irritação',         # Incômodo leve
    'approval': 'aprovação',          # Concordância ou validação
    'caring': 'cuidado',              # Preocupação e empatia
    'confusion': 'confusão',          # Falta de clareza ou entendimento
    'curiosity': 'curiosidade',       # Interesse em saber mais
    'desire': 'desejo',               # Vontade de ter ou fazer algo
    'disappointment': 'decepção',     # Frustração quando expectativas não são atendidas
    'disapproval': 'desaprovação',    # Discordância ou rejeição
    'disgust': 'nojo',                # Repulsa ou aversão
    'embarrassment': 'vergonha',      # Constrangimento
    'excitement': 'empolgação',       # Entusiasmo e energia positiva
    'fear': 'medo',                   # Temor ou ansiedade sobre perigo
    'gratitude': 'gratidão',          # Agradecimento
    'grief': 'tristeza profunda',     # Dor emocional intensa, luto
    'joy': 'alegria',                 # Felicidade e prazer
    'love': 'amor',                   # Afeto profundo
    'nervousness': 'nervosismo',      # Ansiedade leve, inquietação
    'optimism': 'otimismo',           # Esperança e visão positiva do futuro
    'pride': 'orgulho',               # Satisfação com conquistas próprias ou alheias
    'realization': 'percepção',       # Momento de compreensão ou descoberta
    'relief': 'alívio',               # Sensação de conforto após tensão
    'remorse': 'remorso',             # Arrependimento
    'sadness': 'tristeza',            # Estado melancólico
    'surprise': 'surpresa',           # Reação ao inesperado
    'neutral': 'neutro'               # Sem emoção específica
}


# ============================================================================
# FUNÇÃO DE INICIALIZAÇÃO DO MODELO
# ============================================================================

def inicializar_modelo():
    """
    Inicializa o modelo de classificação de emoções GoEmotions.

    Usa lazy loading (carregamento preguiçoso) para carregar o modelo apenas
    quando necessário, não na importação do módulo. Isso melhora o tempo de
    startup do servidor.

    Returns:
        pipeline: Objeto pipeline do Transformers configurado para classificação
    """
    global classificador  # Acessa a variável global para armazenar o modelo

    # Verifica se o modelo já foi carregado anteriormente (singleton pattern)
    if classificador is None:
        print("Carregando modelo GoEmotions... (pode levar alguns segundos)")

        # Cria um pipeline de classificação de texto usando o modelo RoBERTa
        # fine-tuned no dataset GoEmotions
        classificador = pipeline(
            task="text-classification",  # Tipo de tarefa: classificar texto em categorias
            model="SamLowe/roberta-base-go_emotions",  # Modelo do Hugging Face Hub
            top_k=None  # Retorna todas as 28 emoções com suas probabilidades (não apenas a top-1)
        )
        print("Modelo carregado com sucesso!")

    return classificador


# ============================================================================
# FERRAMENTA 1: ANÁLISE BÁSICA DE SENTIMENTO
# ============================================================================

@servidor_mcp.tool()  # Decorator que registra esta função como uma ferramenta MCP
async def analisar_sentimento(texto: str, top_k: int = 5) -> str:
    """
    Analisa o sentimento de um texto usando o dataset GoEmotions do Google.

    Esta é a ferramenta principal e mais usada. Retorna as principais emoções
    detectadas no texto, ideal para análises rápidas sem sobrecarregar com
    todas as 28 emoções.

    O GoEmotions classifica textos em 28 emoções diferentes (retornadas em português):
    admiração, diversão, raiva, irritação, aprovação, cuidado, confusão,
    curiosidade, desejo, decepção, desaprovação, nojo, vergonha, empolgação,
    medo, gratidão, tristeza profunda, alegria, amor, nervosismo, otimismo,
    orgulho, percepção, alívio, remorso, tristeza, surpresa, neutro.

    Args:
        texto: O texto a ser analisado (string em qualquer idioma, mas melhor em inglês)
        top_k: Número de emoções principais a retornar (padrão: 5, máximo: 28)

    Returns:
        JSON string com as emoções detectadas em português e suas probabilidades
    """
    # Inicializa o modelo (ou recupera da cache se já foi inicializado)
    modelo = inicializar_modelo()

    # Executa a inferência do modelo no texto
    # O modelo retorna uma lista de listas, pegamos [0] porque enviamos apenas 1 texto
    resultados = modelo(texto)[0]

    # Ordena as emoções por probabilidade (score) em ordem decrescente
    # lambda x: x['score'] extrai o valor do score de cada emoção para ordenação
    resultados_ordenados = sorted(resultados, key=lambda x: x['score'], reverse=True)

    # Slice para pegar apenas as top_k emoções mais prováveis
    top_resultados = resultados_ordenados[:top_k]

    # Constrói o dicionário de resposta com informações estruturadas
    resposta = {
        "texto_analisado": texto,  # Echo do texto original
        "total_emocoes": len(resultados),  # Sempre 28 para GoEmotions

        # List comprehension para formatar cada emoção top
        "top_emocoes": [
            {
                "emocao": TRADUCAO_EMOCOES.get(r['label'], r['label']),  # Tradução (fallback para original se não encontrar)
                "emocao_original": r['label'],  # Nome em inglês para referência
                "probabilidade": round(r['score'] * 100, 2),  # Score como número (0-100)
                "porcentagem": f"{round(r['score'] * 100, 2)}%"  # Score como string formatada
            }
            for r in top_resultados
        ],

        # Informações sobre a emoção dominante (a de maior probabilidade)
        "emocao_dominante": TRADUCAO_EMOCOES.get(top_resultados[0]['label'], top_resultados[0]['label']),
        "emocao_dominante_original": top_resultados[0]['label'],
        "confianca_dominante": f"{round(top_resultados[0]['score'] * 100, 2)}%"
    }

    # Serializa para JSON com:
    # - indent=2: formatação bonita com indentação
    # - ensure_ascii=False: permite caracteres UTF-8 (acentos em português)
    return json.dumps(resposta, indent=2, ensure_ascii=False)


# ============================================================================
# FERRAMENTA 2: ANÁLISE DETALHADA DE SENTIMENTO
# ============================================================================

@servidor_mcp.tool()  # Decorator que registra esta função como uma ferramenta MCP
async def analisar_sentimento_detalhado(texto: str) -> str:
    """
    Analisa o sentimento de um texto retornando TODAS as 28 emoções com suas probabilidades.

    Esta ferramenta é mais completa que analisar_sentimento, retornando um panorama
    completo de todas as emoções detectadas, agrupadas por níveis de confiança.
    Útil para análises profundas ou quando você quer ver o espectro emocional completo.

    Args:
        texto: O texto a ser analisado

    Returns:
        JSON string com todas as emoções detectadas, suas probabilidades e agrupamentos
    """
    # Inicializa o modelo (ou recupera da cache)
    modelo = inicializar_modelo()

    # Executa a inferência
    resultados = modelo(texto)[0]

    # Ordena por score (probabilidade) decrescente
    resultados_ordenados = sorted(resultados, key=lambda x: x['score'], reverse=True)

    # Agrupa emoções em três categorias baseadas na probabilidade
    # - Alta (>=50%): emoções muito presentes no texto
    # - Média (10-50%): emoções moderadamente presentes
    # - Baixa (<10%): emoções fracamente presentes
    alta_probabilidade = [r for r in resultados_ordenados if r['score'] >= 0.5]
    media_probabilidade = [r for r in resultados_ordenados if 0.1 <= r['score'] < 0.5]
    baixa_probabilidade = [r for r in resultados_ordenados if r['score'] < 0.1]

    # Constrói resposta estruturada
    resposta = {
        "texto_analisado": texto,

        # Informações sobre a emoção dominante
        "emocao_dominante": TRADUCAO_EMOCOES.get(resultados_ordenados[0]['label'], resultados_ordenados[0]['label']),
        "emocao_dominante_original": resultados_ordenados[0]['label'],
        "confianca_dominante": f"{round(resultados_ordenados[0]['score'] * 100, 2)}%",

        # Resumo estatístico da distribuição de emoções
        "resumo": {
            "emocoes_alta_confianca": len(alta_probabilidade),    # Quantas emoções com score >= 50%
            "emocoes_media_confianca": len(media_probabilidade),  # Quantas entre 10-50%
            "emocoes_baixa_confianca": len(baixa_probabilidade)   # Quantas < 10%
        },

        # Lista completa de todas as 28 emoções, ordenadas por probabilidade
        "todas_emocoes": [
            {
                "emocao": TRADUCAO_EMOCOES.get(r['label'], r['label']),
                "emocao_original": r['label'],
                "probabilidade": round(r['score'] * 100, 2),
                "porcentagem": f"{round(r['score'] * 100, 2)}%",
                # Classificação ternária do nível de confiança
                "nivel": "alta" if r['score'] >= 0.5 else "média" if r['score'] >= 0.1 else "baixa"
            }
            for r in resultados_ordenados
        ]
    }

    # Retorna JSON formatado com suporte a UTF-8
    return json.dumps(resposta, indent=2, ensure_ascii=False)


# ============================================================================
# FERRAMENTA 3: COMPARAÇÃO DE SENTIMENTOS
# ============================================================================

@servidor_mcp.tool()  # Decorator que registra esta função como uma ferramenta MCP
async def comparar_sentimentos(textos: list[str]) -> str:
    """
    Compara os sentimentos de múltiplos textos lado a lado.

    Esta ferramenta permite analisar vários textos de uma vez e comparar
    suas emoções dominantes. Útil para:
    - Comparar versões diferentes de um texto (ex: antes/depois de edição)
    - Analisar respostas de diferentes usuários
    - Avaliar mudanças de tom ao longo de uma conversa

    Args:
        textos: Lista de textos a serem analisados e comparados
                Exemplo: ["Estou muito feliz!", "Que dia horrível", "Interessante..."]

    Returns:
        JSON string com análise comparativa de todos os textos
    """
    # Inicializa o modelo
    modelo = inicializar_modelo()

    # Lista para armazenar as análises de cada texto
    analises = []

    # Itera sobre cada texto com índice começando em 1 (mais amigável ao usuário)
    for idx, texto in enumerate(textos, 1):
        # Executa a inferência para este texto
        resultados = modelo(texto)[0]

        # Ordena emoções por probabilidade
        resultados_ordenados = sorted(resultados, key=lambda x: x['score'], reverse=True)

        # Adiciona análise deste texto à lista
        analises.append({
            "texto_numero": idx,  # Número sequencial do texto (1, 2, 3...)
            "texto": texto,  # O texto original

            # Informações sobre a emoção mais forte
            "emocao_dominante": TRADUCAO_EMOCOES.get(resultados_ordenados[0]['label'], resultados_ordenados[0]['label']),
            "emocao_dominante_original": resultados_ordenados[0]['label'],
            "confianca": f"{round(resultados_ordenados[0]['score'] * 100, 2)}%",

            # Top 3 emoções para dar contexto adicional
            # (útil quando a dominante não é tão forte ou há emoções mistas)
            "top_3_emocoes": [
                {
                    "emocao": TRADUCAO_EMOCOES.get(r['label'], r['label']),
                    "emocao_original": r['label'],
                    "probabilidade": f"{round(r['score'] * 100, 2)}%"
                }
                for r in resultados_ordenados[:3]  # Slice para pegar apenas as 3 primeiras
            ]
        })

    # Estrutura a resposta final com metadados e todas as análises
    resposta = {
        "total_textos_analisados": len(textos),  # Quantos textos foram processados
        "analises": analises  # Lista com análise individual de cada texto
    }

    # Retorna JSON formatado
    return json.dumps(resposta, indent=2, ensure_ascii=False)


# ============================================================================
# PONTO DE ENTRADA DO PROGRAMA
# ============================================================================

if __name__ == "__main__":
    """
    Este bloco só é executado quando o script é rodado diretamente
    (não quando é importado como módulo).

    Fluxo de inicialização:
    1. Carrega o modelo GoEmotions na memória
    2. Inicia o servidor MCP na porta 8080
    3. Fica aguardando requisições dos clientes MCP
    """

    # Pré-carrega o modelo antes de aceitar conexões
    # Isso evita delay na primeira requisição (que seria mais lenta)
    inicializar_modelo()

    # Inicia o servidor MCP usando Server-Sent Events (SSE)
    # - transport='sse': usa SSE para comunicação (streaming unidirecional do servidor)
    # - port=8080: porta onde o servidor ficará escutando
    # O servidor ficará rodando até ser interrompido (Ctrl+C)
    servidor_mcp.run(transport='sse', port=8080)
