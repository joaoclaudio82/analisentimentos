import asyncio
import json
import os

import dotenv
from fastmcp import Client
from openai import OpenAI

caminho_servidor = 'http://localhost:8080/sse'
cliente_mcp = Client(caminho_servidor)


async def exemplo_analise_basica():
    """Exemplo 1: Análise básica de sentimento (top 5 emoções)"""
    print("\n" + "="*80)
    print("EXEMPLO 1: Análise Básica de Sentimento")
    print("="*80 + "\n")

    texto = "Estou muito feliz e animado com essa nova oportunidade! Mal posso esperar para começar!"

    async with cliente_mcp:
        resultado = await cliente_mcp.call_tool(
            "analisar_sentimento",
            arguments={'texto': texto, 'top_k': 5}
        )
        print(resultado[0].text)


async def exemplo_analise_detalhada():
    """Exemplo 2: Análise detalhada (todas as 28 emoções)"""
    print("\n" + "="*80)
    print("EXEMPLO 2: Análise Detalhada (Todas as 28 Emoções)")
    print("="*80 + "\n")

    texto = "Estou preocupado com o futuro, mas também esperançoso de que tudo vai dar certo."

    async with cliente_mcp:
        resultado = await cliente_mcp.call_tool(
            "analisar_sentimento_detalhado",
            arguments={'texto': texto}
        )
        print(resultado[0].text)


async def exemplo_comparacao():
    """Exemplo 3: Comparação de múltiplos textos"""
    print("\n" + "="*80)
    print("EXEMPLO 3: Comparação de Sentimentos")
    print("="*80 + "\n")

    textos = [
        "Que dia maravilhoso! Tudo está perfeito!",
        "Estou muito frustrado e irritado com essa situação.",
        "Não sei o que pensar sobre isso, estou confuso.",
        "Obrigado por tudo! Você é incrível!"
    ]

    async with cliente_mcp:
        resultado = await cliente_mcp.call_tool(
            "comparar_sentimentos",
            arguments={'textos': textos}
        )
        print(resultado[0].text)


async def exemplo_com_openai():
    """Exemplo 4: Integração com OpenAI para análise contextual"""
    print("\n" + "="*80)
    print("EXEMPLO 4: Análise com Síntese OpenAI")
    print("="*80 + "\n")

    dotenv.load_dotenv()
    api_key = os.environ.get('CHAVE_API_OPENAI')

    if not api_key:
        print("⚠️  CHAVE_API_OPENAI não encontrada. Pulando exemplo com OpenAI.")
        return

    texto = """
    Recebi a notícia de que fui aprovado no emprego dos sonhos!
    Estou nas nuvens, mas também um pouco nervoso com os novos desafios.
    """

    async with cliente_mcp:
        # Análise de sentimento
        resultado = await cliente_mcp.call_tool(
            "analisar_sentimento",
            arguments={'texto': texto, 'top_k': 5}
        )

        analise_sentimento = resultado[0].text

        print("📊 Análise de Sentimento:")
        print(analise_sentimento)

        # Síntese com OpenAI
        mensagem_sistema = f"""
        Você é um assistente especializado em análise emocional.
        Um usuário escreveu o seguinte texto: "{texto}"

        A análise de sentimentos GoEmotions detectou as seguintes emoções:
        {analise_sentimento}

        Com base nessa análise, forneça:
        1. Uma interpretação do estado emocional da pessoa
        2. Insights sobre o que ela pode estar vivenciando
        3. Sugestões de como ela pode processar essas emoções

        Seja empático e construtivo.
        """

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=mensagem_sistema,
            input="Por favor, analise meu estado emocional e me dê algumas orientações.",
        )

        print("\n🤖 Síntese e Orientação (OpenAI):")
        print("-" * 80)
        print(response.output_text)


async def exemplo_analise_review():
    """Exemplo 5: Análise de review de produto"""
    print("\n" + "="*80)
    print("EXEMPLO 5: Análise de Review de Produto")
    print("="*80 + "\n")

    review = """
    Comprei este produto há uma semana e estou muito decepcionado.
    A qualidade é péssima e não funciona como prometido.
    Me sinto enganado e frustrado. Não recomendo!
    """

    async with cliente_mcp:
        resultado = await cliente_mcp.call_tool(
            "analisar_sentimento_detalhado",
            arguments={'texto': review}
        )

        print("📝 Review analisado:")
        print(review)
        print("\n📊 Análise de Emoções:")
        print(resultado[0].text)


async def main():
    """Executa todos os exemplos"""
    print("\n" + "🎭" * 40)
    print("ANÁLISE DE SENTIMENTOS COM GoEmotions (28 Emoções)")
    print("🎭" * 40)

    # Executa os exemplos
    await exemplo_analise_basica()
    await exemplo_analise_detalhada()
    await exemplo_comparacao()
    await exemplo_analise_review()
    await exemplo_com_openai()

    print("\n" + "="*80)
    print("✅ Todos os exemplos executados com sucesso!")
    print("="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(main())
