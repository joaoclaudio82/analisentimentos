# Análise de Sentimentos com MCP e GoEmotions

Projeto didático que expõe um servidor [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) para classificação de emoções com o modelo **GoEmotions** (28 categorias). Inclui um cliente de exemplo que consome as ferramentas MCP diretamente e integra a análise com um resumo gerado pela API da OpenAI.

## Visão Geral

- `servidor_sentimentos.py`: servidor MCP com três ferramentas (`analisar_sentimento`, `analisar_sentimento_detalhado`, `comparar_sentimentos`).
- `cliente.py`: exemplos assíncronos que consomem o servidor via `fastmcp.Client`, incluindo um fluxo opcional com OpenAI para síntese contextual.
- `paper.md`: material de apoio com explicações e exemplos de interpretação de emoções.

## Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv) (recomendado) ou `pip`
- Conta na OpenAI (opcional, apenas para o exemplo que gera recomendações)  
  Defina a variável de ambiente `CHAVE_API_OPENAI` ou crie um arquivo `.env` com:
  ```
  CHAVE_API_OPENAI="sua-chave-aqui"
  ```

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/joaoclaudio82/analisentimentos.git
cd analisentimentos

# Criar ambiente virtual e instalar dependências (com uv)
uv sync

# Ativar o ambiente
source .venv/bin/activate
```

> Prefere `pip`? Use `python -m venv .venv`, ative o ambiente e rode `pip install -r requirements.txt` gerado por `uv pip compile pyproject.toml`.

## Como Executar

1. **Iniciar o servidor MCP**
   ```bash
   python servidor_sentimentos.py
   ```
   O servidor carrega o modelo `SamLowe/roberta-base-go_emotions` e disponibiliza as ferramentas em `http://localhost:8080/sse`.

2. **Rodar o cliente de demonstração (opcional)**
   Em outro terminal, com o ambiente ativado:
   ```bash
   python cliente.py
   ```
   O script executa cinco exemplos:
   - Análise básica (top 5 emoções)
   - Análise detalhada (todas as 28 emoções)
   - Comparação de múltiplos textos
   - Análise de review de produto
   - Fluxo combinado com a API da OpenAI (requer `CHAVE_API_OPENAI`)

## Ferramentas Disponíveis

| Ferramenta | Descrição | Uso principal |
|------------|-----------|---------------|
| `analisar_sentimento` | Retorna as top *k* emoções traduzidas | Dashboards, respostas rápidas |
| `analisar_sentimento_detalhado` | Lista completa das 28 emoções, com níveis de confiança | Análises profundas, relatórios |
| `comparar_sentimentos` | Contrasta emoções dominantes em uma lista de textos | Monitoramento de conversas, experimentos A/B |

Todas as respostas são JSON com rótulos em português via dicionário de tradução.

## Estrutura do Projeto

```
├── cliente.py
├── paper.md
├── pyproject.toml
├── servidor_sentimentos.py
├── uv.lock
└── README.md
```

## Roadmap

- [ ] Adicionar testes automatizados para as ferramentas MCP
- [ ] Containerizar servidor e cliente com Docker
- [ ] Expor API REST/GraphQL alternativa
- [ ] Criar interface web simples para visualização das emoções

## Licença

Defina a licença desejada (MIT, Apache 2.0 etc.) e adicione o arquivo correspondente. Atualmente o projeto não possui licença explícita.

