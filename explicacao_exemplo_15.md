# Explicação do Exemplo 15 - Análise de Sentimentos com GoEmotions

## O que é GoEmotions?

**GoEmotions** é um dataset de pesquisa desenvolvido pelo Google Research que contém 58.000 comentários do Reddit anotados manualmente com 28 categorias de emoções diferentes. Este é o maior dataset de análise de emoções granulares disponível publicamente.

### Diferencial do GoEmotions

Ao contrário de análises tradicionais de sentimento que classificam texto apenas como:
- ✅ Positivo
- ❌ Negativo
- ➖ Neutro

O GoEmotions oferece uma análise **muito mais rica e detalhada** com **28 emoções diferentes**, permitindo capturar nuances complexas do estado emocional humano.

### As 28 Emoções Classificadas

**Positivas (12):**
1. admiration (admiração)
2. amusement (diversão)
3. approval (aprovação)
4. caring (cuidado)
5. desire (desejo)
6. excitement (empolgação)
7. gratitude (gratidão)
8. joy (alegria)
9. love (amor)
10. optimism (otimismo)
11. pride (orgulho)
12. relief (alívio)

**Negativas (11):**
13. anger (raiva)
14. annoyance (irritação)
15. disappointment (decepção)
16. disapproval (desaprovação)
17. disgust (nojo)
18. embarrassment (vergonha)
19. fear (medo)
20. grief (tristeza profunda)
21. nervousness (nervosismo)
22. remorse (remorso)
23. sadness (tristeza)

**Ambíguas (4):**
24. confusion (confusão)
25. curiosity (curiosidade)
26. realization (percepção)
27. surprise (surpresa)

**Neutra (1):**
28. neutral (neutro)

---

## Explicação Detalhada do Arquivo `servidor_sentimentos.py`

### Visão Geral

Este servidor MCP integra um modelo de Machine Learning (RoBERTa) pré-treinado no dataset GoEmotions para fornecer análise avançada de sentimentos através de três ferramentas diferentes.

### Importações e Configuração Inicial

```python
import json
from transformers import pipeline
from fastmcp import FastMCP
```

**Explicação:**
- **`json`**: Para formatar as respostas em JSON estruturado
- **`transformers`**: Biblioteca HuggingFace que fornece acesso fácil a modelos pré-treinados
- **`pipeline`**: Abstração de alto nível que simplifica o uso de modelos de ML
- **`FastMCP`**: Framework para criar servidores MCP

### Inicialização do Servidor e Modelo

```python
servidor_mcp = FastMCP('mcp-analise-sentimentos')
classificador = None

def inicializar_modelo():
    global classificador
    if classificador is None:
        print("Carregando modelo GoEmotions... (pode levar alguns segundos)")
        classificador = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None
        )
        print("Modelo carregado com sucesso!")
    return classificador
```

**Explicação:**
- **`classificador = None`**: Variável global que armazenará o modelo
- **`inicializar_modelo()`**: Função que carrega o modelo uma única vez (lazy loading)
- **`model="SamLowe/roberta-base-go_emotions"`**: Modelo RoBERTa pré-treinado no dataset GoEmotions disponível no HuggingFace
- **`top_k=None`**: Retorna todas as 28 emoções com suas probabilidades
- **Benefício do lazy loading**: O modelo (~500MB) só é baixado e carregado quando realmente necessário

### Ferramenta 1: Análise Básica de Sentimento

```python
@servidor_mcp.tool()
async def analisar_sentimento(texto: str, top_k: int = 5) -> str:
    modelo = inicializar_modelo()
    resultados = modelo(texto)[0]
    resultados_ordenados = sorted(resultados, key=lambda x: x['score'], reverse=True)
    top_resultados = resultados_ordenados[:top_k]

    resposta = {
        "texto_analisado": texto,
        "total_emocoes": len(resultados),
        "top_emocoes": [
            {
                "emocao": r['label'],
                "probabilidade": round(r['score'] * 100, 2),
                "porcentagem": f"{round(r['score'] * 100, 2)}%"
            }
            for r in top_resultados
        ],
        "emocao_dominante": top_resultados[0]['label'],
        "confianca_dominante": f"{round(top_resultados[0]['score'] * 100, 2)}%"
    }

    return json.dumps(resposta, indent=2, ensure_ascii=False)
```

**Explicação Detalhada:**

1. **`@servidor_mcp.tool()`**: Registra a função como ferramenta MCP
2. **`modelo = inicializar_modelo()`**: Garante que o modelo está carregado
3. **`resultados = modelo(texto)[0]`**: Executa a inferência do modelo
   - Retorna lista de dicionários: `[{'label': 'joy', 'score': 0.95}, ...]`
4. **`sorted(..., key=lambda x: x['score'], reverse=True)`**: Ordena por probabilidade (maior para menor)
5. **`top_resultados = resultados_ordenados[:top_k]`**: Pega apenas as top K emoções
6. **List comprehension**: Formata cada emoção com seus dados
7. **`json.dumps(..., ensure_ascii=False)`**: Converte para JSON preservando caracteres UTF-8

**Quando usar:**
- Análise rápida de texto
- Identificar as principais emoções presentes
- Dashboards e relatórios resumidos

### Ferramenta 2: Análise Detalhada

```python
@servidor_mcp.tool()
async def analisar_sentimento_detalhado(texto: str) -> str:
    modelo = inicializar_modelo()
    resultados = modelo(texto)[0]
    resultados_ordenados = sorted(resultados, key=lambda x: x['score'], reverse=True)

    alta_probabilidade = [r for r in resultados_ordenados if r['score'] >= 0.5]
    media_probabilidade = [r for r in resultados_ordenados if 0.1 <= r['score'] < 0.5]
    baixa_probabilidade = [r for r in resultados_ordenados if r['score'] < 0.1]

    resposta = {
        "texto_analisado": texto,
        "emocao_dominante": resultados_ordenados[0]['label'],
        "confianca_dominante": f"{round(resultados_ordenados[0]['score'] * 100, 2)}%",
        "resumo": {
            "emocoes_alta_confianca": len(alta_probabilidade),
            "emocoes_media_confianca": len(media_probabilidade),
            "emocoes_baixa_confianca": len(baixa_probabilidade)
        },
        "todas_emocoes": [...]
    }

    return json.dumps(resposta, indent=2, ensure_ascii=False)
```

**Explicação:**
- **Agrupamento por confiança**:
  - Alta: ≥ 50% (emoções fortemente presentes)
  - Média: 10-49% (emoções moderadamente presentes)
  - Baixa: < 10% (emoções fracamente presentes)
- **`todas_emocoes`**: Retorna as 28 emoções com suas probabilidades
- **Nível de confiança**: Adiciona classificação qualitativa ("alta", "média", "baixa")

**Quando usar:**
- Análise profunda de textos complexos
- Pesquisa e estudos acadêmicos
- Entender todas as nuances emocionais

### Ferramenta 3: Comparação de Sentimentos

```python
@servidor_mcp.tool()
async def comparar_sentimentos(textos: list[str]) -> str:
    modelo = inicializar_modelo()

    analises = []
    for idx, texto in enumerate(textos, 1):
        resultados = modelo(texto)[0]
        resultados_ordenados = sorted(resultados, key=lambda x: x['score'], reverse=True)

        analises.append({
            "texto_numero": idx,
            "texto": texto,
            "emocao_dominante": resultados_ordenados[0]['label'],
            "confianca": f"{round(resultados_ordenados[0]['score'] * 100, 2)}%",
            "top_3_emocoes": [...]
        })

    return json.dumps({"total_textos_analisados": len(textos), "analises": analises}, ...)
```

**Explicação:**
- **`list[str]`**: Aceita múltiplos textos como entrada
- **`enumerate(textos, 1)`**: Numera os textos começando de 1
- **Loop de análise**: Processa cada texto individualmente
- **Comparação estruturada**: Permite comparar emoções lado a lado

**Quando usar:**
- Comparar variações de texto (A/B testing)
- Analisar evolução emocional em conversas
- Benchmarking de diferentes versões de conteúdo

---

## Explicação Detalhada do Arquivo `15_cliente.py`

### Exemplo 1: Análise Básica

```python
async def exemplo_analise_basica():
    texto = "Estou muito feliz e animado com essa nova oportunidade!"

    async with cliente_mcp:
        resultado = await cliente_mcp.call_tool(
            "analisar_sentimento",
            arguments={'texto': texto, 'top_k': 5}
        )
        print(resultado.content[0].text)
```

**Saída esperada:**
```json
{
  "texto_analisado": "Estou muito feliz e animado com essa nova oportunidade!",
  "total_emocoes": 28,
  "top_emocoes": [
    {"emocao": "joy", "probabilidade": 89.23, "porcentagem": "89.23%"},
    {"emocao": "excitement", "probabilidade": 78.45, "porcentagem": "78.45%"},
    {"emocao": "optimism", "probabilidade": 65.12, "porcentagem": "65.12%"},
    {"emocao": "approval", "probabilidade": 23.56, "porcentagem": "23.56%"},
    {"emocao": "admiration", "probabilidade": 15.34, "porcentagem": "15.34%"}
  ],
  "emocao_dominante": "joy",
  "confianca_dominante": "89.23%"
}
```

### Exemplo 2: Análise Detalhada

Mostra todas as 28 emoções com classificação por nível de confiança.

### Exemplo 3: Comparação de Textos

Compara 4 textos diferentes com emoções contrastantes.

### Exemplo 4: Integração com OpenAI

```python
async def exemplo_com_openai():
    # Análise de sentimento com MCP
    resultado = await cliente_mcp.call_tool("analisar_sentimento", ...)

    # Síntese com OpenAI
    mensagem_sistema = f"""
    Você é um assistente especializado em análise emocional.
    Um usuário escreveu: "{texto}"

    A análise GoEmotions detectou: {analise_sentimento}

    Forneça:
    1. Interpretação do estado emocional
    2. Insights sobre o que a pessoa está vivenciando
    3. Sugestões de como processar essas emoções
    """

    response = client.responses.create(model="gpt-4o-mini", ...)
```

**Explicação:**
- Combina a análise objetiva do GoEmotions com a capacidade de síntese da OpenAI
- Cria um pipeline: Texto → Análise de Emoções → Interpretação Humana
- Demonstra o poder de combinar múltiplas ferramentas MCP

### Exemplo 5: Análise de Review

Demonstra uso prático analisando uma avaliação negativa de produto.

---

## Vantagens da Abordagem MCP com ML

### 1. Separação de Responsabilidades
- **Servidor**: Gerencia o modelo de ML, cache, otimizações
- **Cliente**: Foca na lógica de negócio e apresentação
- **IDE/Claude**: Pode usar a ferramenta sem conhecer detalhes de implementação

### 2. Reutilização
- O mesmo servidor pode ser usado por:
  - Claude Desktop
  - Cursor AI
  - Scripts Python personalizados
  - Qualquer cliente MCP

### 3. Escalabilidade
- Modelo carregado uma vez, usado múltiplas vezes
- Fácil adicionar cache de resultados
- Possibilidade de rodar em servidor remoto

### 4. Manutenção
- Atualizar o modelo: apenas no servidor
- Adicionar features: criar novas ferramentas
- Versioning: múltiplos servidores com diferentes modelos

---

## Como Executar

### 1. Instalar Dependências

```bash
uv sync
# ou
pip install transformers torch fastmcp
```

### 2. Executar o Servidor

```bash
python servidor_sentimentos.py
```

Na primeira execução, o modelo (~500MB) será baixado automaticamente.

### 3. Executar o Cliente (em outro terminal)

```bash
python 15_cliente.py
```

### 4. Usar no Claude Desktop

Copie o conteúdo de `claude_desktop_config.json` para o arquivo de configuração do Claude Desktop.

Depois, você pode simplesmente conversar com Claude:
- "Analise o sentimento deste texto: ..."
- "Compare as emoções destes dois comentários: ..."
- "Faça uma análise detalhada das emoções: ..."

---

## Casos de Uso Práticos

### 1. Análise de Feedback de Clientes
Identificar emoções específicas em reviews e comentários para ações direcionadas.

### 2. Monitoramento de Redes Sociais
Detectar não apenas sentimento negativo, mas emoções específicas como raiva, medo, ou decepção.

### 3. Saúde Mental e Bem-Estar
Análise de textos de diários ou conversas para identificar padrões emocionais.

### 4. Atendimento ao Cliente
Priorizar tickets baseado em emoções detectadas (raiva > frustração > confusão).

### 5. Criação de Conteúdo
A/B testing de textos para otimizar respostas emocionais desejadas.

### 6. Pesquisa de Mercado
Análise qualitativa de respostas abertas em pesquisas.

---

## Limitações e Considerações

### 1. Idioma
- O modelo foi treinado em inglês
- Para português, pode haver perda de precisão
- Considere usar tradutor ou modelo multilíngue

### 2. Contexto
- Ironia e sarcasmo podem ser mal interpretados
- Gírias e expressões regionais podem não ser reconhecidas

### 3. Performance
- Primeira execução: download do modelo (~500MB)
- Tempo de inferência: ~100-500ms por texto
- Uso de memória: ~1-2GB RAM

### 4. Multi-label
- Textos podem ter múltiplas emoções simultaneamente
- A "dominante" nem sempre conta a história completa

---

## Melhorias Futuras

1. **Cache de resultados**: Armazenar análises de textos já processados
2. **Batch processing**: Processar múltiplos textos em paralelo
3. **Modelo multilíngue**: Usar modelo treinado em português
4. **Visualizações**: Gráficos das distribuições emocionais
5. **Análise temporal**: Rastrear mudanças emocionais ao longo do tempo
6. **Fine-tuning**: Treinar modelo em domínio específico

---

## Recursos Adicionais

- **Paper Original**: [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
- **HuggingFace Model**: [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- **Dataset**: [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Transformers Docs**: [HuggingFace Transformers](https://huggingface.co/docs/transformers)
