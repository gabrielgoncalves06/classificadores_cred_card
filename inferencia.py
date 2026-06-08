import pandas as pd
import joblib
import json


modelo = joblib.load('modelo_final.pkl')
print("Modelo carregado com sucesso!")


# 2. Função principal de inferência
def prever_risco_cliente(dados_cliente):
    # Converte os dados do cliente (dicionário) para uma tabela de 1 linha
    df_cliente = pd.DataFrame([dados_cliente])

    # Faz a previsão binária (0 = Bom Pagador, 1 = Inadimplente)
    previsao = modelo.predict(df_cliente)[0]

    # Extrai as probabilidades (Score)
    probabilidades = modelo.predict_proba(df_cliente)[0]
    chance_bom_pagador = probabilidades[0] * 100
    chance_inadimplente = probabilidades[1] * 100

    # 3. Retorna o resultado formatado
    return {
        "Risco_Identificado": "Inadimplente (Default)" if previsao == 1 else "Bom Pagador",
        "Score_de_Inadimplencia": f"{chance_inadimplente:.1f}%",
        "Distribuicao_de_Probabilidade": {
            "Chance_de_Pagar": f"{chance_bom_pagador:.1f}%",
            "Chance_de_Nao_Pagar": f"{chance_inadimplente:.1f}%"
        }
    }


# --- Teste do Sistema com dados reais do seu dataset ---
if __name__ == "__main__":
    # Usando os dados da linha 5 do seu CSV como exemplo (cliente ID 4)
    cliente_teste = {
        'LIMIT_BAL': 50000,
        'SEX': 'M',  # M = Masculino
        'EDUCATION': 'Middle School',
        'MARRIAGE': 'Married',
        'AGE': 37,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 46990,
        'BILL_AMT2': 48233,
        'BILL_AMT3': 49291,
        'BILL_AMT4': 28314,
        'BILL_AMT5': 28959,
        'BILL_AMT6': 29547,
        'PAY_AMT1': 2000,
        'PAY_AMT2': 2019,
        'PAY_AMT3': 1200,
        'PAY_AMT4': 1100,
        'PAY_AMT5': 1069,
        'PAY_AMT6': 1000
    }

    print("\n=== TESTE DE INFERÊNCIA ===\n")
    print(
        f"Dados do cliente: {cliente_teste['SEX']}, {cliente_teste['EDUCATION']}, {cliente_teste['MARRIAGE']}, {cliente_teste['AGE']} anos")
    print(f"Limite: R$ {cliente_teste['LIMIT_BAL']:.0f}")
    print()

    resultado = prever_risco_cliente(cliente_teste)
    print(json.dumps(resultado, indent=4, ensure_ascii=False))

    # Teste adicional com outro perfil
    print("\n" + "=" * 50)
    print("TESTE COM CLIENTE INADIMPLENTE (Linha 2 do CSV):")
    print("=" * 50)

    cliente_inadimplente = {
        'LIMIT_BAL': 120000,
        'SEX': 'M',
        'EDUCATION': 'Middle School',
        'MARRIAGE': 'Widowed',
        'AGE': 26,
        'PAY_0': -1,
        'PAY_2': 2,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 2,
        'BILL_AMT1': 2682,
        'BILL_AMT2': 1725,
        'BILL_AMT3': 2682,
        'BILL_AMT4': 3272,
        'BILL_AMT5': 3455,
        'BILL_AMT6': 3261,
        'PAY_AMT1': 0,
        'PAY_AMT2': 1000,
        'PAY_AMT3': 1000,
        'PAY_AMT4': 1000,
        'PAY_AMT5': 0,
        'PAY_AMT6': 2000
    }

    resultado2 = prever_risco_cliente(cliente_inadimplente)
    print(json.dumps(resultado2, indent=4, ensure_ascii=False))