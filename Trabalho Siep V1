import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import binom
import matplotlib.pyplot as plt
import math

# Configuração da página
st.set_page_config(page_title="Análise de Operações Aéreas - Aérea Confiável", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# Título e introdução
st.title("Análise de Operações Aéreas - Aérea Confiável")
st.markdown("""
Este aplicativo apresenta análises para apoiar decisões estratégicas da companhia aérea fictícia
Aérea Confiável, com foco em gestão de assentos e retorno sobre investimentos em tecnologia.
""")

# Sidebar com parâmetros ajustáveis
st.sidebar.header("Parâmetros de Análise")

# Parâmetros de overbooking
st.sidebar.subheader("Parâmetros de Overbooking")
capacidade_aviao = st.sidebar.number_input("Capacidade do avião (assentos)", value=120, min_value=50, max_value=500)
prob_comparecimento = st.sidebar.slider("Probabilidade de comparecimento (%)", value=88, min_value=70, max_value=100) / 100
max_passagens = st.sidebar.number_input("Máximo de passagens para análise", value=130, min_value=int(capacidade_aviao), max_value=int(capacidade_aviao) + 20)
risco_maximo = st.sidebar.slider("Risco máximo de overbooking aceitável (%)", value=7, min_value=1, max_value=20) / 100

# Parâmetros financeiros
st.sidebar.subheader("Parâmetros Financeiros")
preco_passagem = st.sidebar.number_input("Preço médio da passagem (R$)", value=800, min_value=100, max_value=10000)
custo_compensacao = st.sidebar.number_input("Custo médio de compensação por passageiro (R$)", value=1500, min_value=500, max_value=5000)
custo_reputacao = st.sidebar.number_input("Custo estimado de dano à reputação por incidente (R$)", value=5000, min_value=0, max_value=50000)

# Parâmetros de investimento em SI
st.sidebar.subheader("Investimento em Sistema de Informação")
receita_adicional_projetada = st.sidebar.number_input("Receita adicional projetada (R$)", value=80000, min_value=10000, max_value=500000)
custos_operacionais_anuais = st.sidebar.number_input("Custos operacionais anuais (R$)", value=10000, min_value=1000, max_value=100000)
custo_investimento = st.sidebar.number_input("Custo do investimento (R$)", value=100000, min_value=10000, max_value=1000000)

# Parâmetros de simulação
st.sidebar.subheader("Parâmetros de Simulação Monte Carlo")
num_simulacoes = st.sidebar.slider("Número de simulações", value=1000, min_value=100, max_value=10000)
variacao_receita = st.sidebar.slider("Variação da receita (%)", value=20, min_value=5, max_value=50)
variacao_custos = st.sidebar.slider("Variação dos custos (%)", value=15, min_value=5, max_value=50)

# Divisão em abas
tab1, tab2, tab3, tab4 = st.tabs(["Análise de Overbooking", "Viabilidade Financeira", "ROI do Sistema de Informação", "Simulações Monte Carlo"])

# Tab 1: Análise de Overbooking
with tab1:
    st.header("Análise de Probabilidade de Overbooking")
    
    # Descrição da análise
    st.markdown("""
    Esta análise utiliza a distribuição binomial para calcular a probabilidade de overbooking em voos.
    Um overbooking ocorre quando o número de passageiros que comparecem excede a capacidade de assentos do avião.
    """)
    
    # Função para calcular probabilidade de overbooking
    @st.cache_data
    def calcular_prob_overbooking(n_passagens, capacidade, prob_comparecimento):
        # Probabilidade de k ou menos passageiros comparecerem (k = capacidade)
        prob_sem_overbooking = binom.cdf(capacidade, n_passagens, prob_comparecimento)
        # Probabilidade de mais de k passageiros comparecerem = overbooking
        prob_overbooking = 1 - prob_sem_overbooking
        return prob_overbooking
    
    # Calculando probabilidades para diferentes números de passagens vendidas
    range_passagens = range(int(capacidade_aviao), int(max_passagens) + 1)
    prob_overbooking = [calcular_prob_overbooking(n, capacidade_aviao, prob_comparecimento) * 100 for n in range_passagens]
    
    # Encontrar o número máximo de passagens que mantém o risco abaixo do limite
    df_prob = pd.DataFrame({
        "Passagens Vendidas": range_passagens,
        "Probabilidade de Overbooking (%)": prob_overbooking
    })
    
    max_passagens_seguras = 0
    for i, row in df_prob.iterrows():
        if row["Probabilidade de Overbooking (%)"] <= risco_maximo * 100:
            max_passagens_seguras = row["Passagens Vendidas"]
    
    # Criando gráfico interativo
    fig = px.line(df_prob, x="Passagens Vendidas", y="Probabilidade de Overbooking (%)", 
                  title="Probabilidade de Overbooking por Número de Passagens Vendidas")
    
    # Adicionar linha horizontal para o limite de risco
    fig.add_hline(y=risco_maximo * 100, line_dash="dash", line_color="red", 
                  annotation_text=f"Limite de risco ({risco_maximo*100}%)")
    
    # Destacar o número máximo de passagens seguras
    if max_passagens_seguras > 0:
        fig.add_vline(x=max_passagens_seguras, line_dash="dash", line_color="green",
                      annotation_text=f"Máx. passagens seguras ({max_passagens_seguras})")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar resultado do número máximo de passagens seguras
    if max_passagens_seguras > 0:
        st.success(f"O número máximo de passagens que podem ser vendidas mantendo o risco de overbooking abaixo de {risco_maximo*100}% é: **{max_passagens_seguras}**")
    else:
        st.warning(f"Não é possível manter o risco abaixo de {risco_maximo*100}% com os parâmetros atuais.")
    
    # Detalhamento das probabilidades
    st.subheader("Detalhamento das Probabilidades")
    st.dataframe(df_prob)
    
    # Exemplo de cálculo específico
    st.subheader("Demonstração de Cálculo")
    n_exemplo = 130
    st.markdown(f"""
    **Exemplo de cálculo para {n_exemplo} passagens vendidas:**
    
    - Capacidade do avião: {capacidade_aviao} assentos
    - Probabilidade de comparecimento: {prob_comparecimento*100}%
    - Overbooking ocorre quando mais de {capacidade_aviao} passageiros comparecem
    
    Usando a distribuição binomial:
    - P(X > {capacidade_aviao}) = 1 - P(X ≤ {capacidade_aviao})
    - P(X ≤ {capacidade_aviao}) = soma das probabilidades de 0 a {capacidade_aviao} passageiros comparecerem
    """)
    
    prob_específica = calcular_prob_overbooking(n_exemplo, capacidade_aviao, prob_comparecimento) * 100
    st.markdown(f"**Resultado:** A probabilidade de overbooking com {n_exemplo} passagens vendidas é de **{prob_específica:.2f}%**")

# Tab 2: Viabilidade Financeira
with tab2:
    st.header("Análise de Viabilidade Financeira da Venda de Passagens Extras")
    
    st.markdown("""
    Esta análise avalia se é financeiramente vantajoso vender passagens extras considerando:
    - Receita adicional gerada pelas passagens extras
    - Custos potenciais de compensação em caso de overbooking
    - Custos estimados de danos à imagem/reputação da empresa
    """)
    
    # Função para calcular o valor esperado da venda de passagens extras
    def calcular_valor_esperado(n_passagens_extras, capacidade, prob_comparecimento, preco, custo_comp, custo_rep):
        # Receita total das passagens extras
        receita_extra = n_passagens_extras * preco
        
        # Para cada possível número de passageiros extras que comparecem
        valor_esperado_custo = 0
        for k in range(n_passagens_extras + 1):
            # Probabilidade de exatamente k passageiros extras comparecerem
            prob_k = binom.pmf(k, n_passagens_extras, prob_comparecimento)
            
            # Custo se k > capacidade extra (ou seja, se houver overbooking)
            if k > 0:  # Considerando que já estamos no limite da capacidade
                custo = k * custo_comp + custo_rep
                valor_esperado_custo += prob_k * custo
        
        # Valor esperado = receita - custo esperado
        return receita_extra, valor_esperado_custo, receita_extra - valor_esperado_custo
    
    # Calcular para diferentes números de passagens extras
    n_extras_range = range(1, 21)  # Analisar de 1 a 20 passagens extras
    resultados = []
    
    for n_extras in n_extras_range:
        receita, custo_esperado, valor_liquido = calcular_valor_esperado(
            n_extras, capacidade_aviao, prob_comparecimento, preco_passagem, custo_compensacao, custo_reputacao)
        
        # Probabilidade de overbooking com essas passagens extras
        nova_capacidade = capacidade_aviao + n_extras
        prob_over = calcular_prob_overbooking(nova_capacidade, capacidade_aviao, prob_comparecimento) * 100
        
        resultados.append({
            "Passagens Extras": n_extras,
            "Receita Adicional (R$)": receita,
            "Custo Esperado (R$)": custo_esperado,
            "Valor Líquido Esperado (R$)": valor_liquido,
            "Probabilidade de Overbooking (%)": prob_over
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar foco nas 10 passagens extras solicitadas
    st.subheader("Foco na Venda de 10 Passagens Extras")
    dados_10_extras = df_resultados[df_resultados["Passagens Extras"] == 10].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Receita Adicional", f"R$ {dados_10_extras['Receita Adicional (R$)']:.2f}")
    col2.metric("Custo Esperado", f"R$ {dados_10_extras['Custo Esperado (R$)']:.2f}")
    col3.metric("Valor Líquido", f"R$ {dados_10_extras['Valor Líquido Esperado (R$)']:.2f}")
    col4.metric("Prob. Overbooking", f"{dados_10_extras['Probabilidade de Overbooking (%)']:.2f}%")
    
    # Conclusão sobre a venda de 10 passagens extras
    if dados_10_extras["Valor Líquido Esperado (R$)"] > 0:
        st.success("É financeiramente vantajoso vender 10 passagens extras, com um valor líquido esperado positivo.")
    else:
        st.error("Não é financeiramente vantajoso vender 10 passagens extras, pois o valor líquido esperado é negativo.")
    
    # Gráfico comparativo para diferentes números de passagens extras
    st.subheader("Análise Comparativa para Diferentes Números de Passagens Extras")
    
    fig1 = px.bar(df_resultados, x="Passagens Extras", y=["Receita Adicional (R$)", "Custo Esperado (R$)"],
                 title="Receita vs. Custo Esperado por Número de Passagens Extras",
                 barmode="group")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.line(df_resultados, x="Passagens Extras", y="Valor Líquido Esperado (R$)",
                  title="Valor Líquido Esperado por Número de Passagens Extras")
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Identificar o número ótimo de passagens extras
    idx_otimo = df_resultados["Valor Líquido Esperado (R$)"].idxmax()
    n_otimo = df_resultados.iloc[idx_otimo]["Passagens Extras"]
    valor_otimo = df_resultados.iloc[idx_otimo]["Valor Líquido Esperado (R$)"]
    
    st.success(f"O número ótimo de passagens extras a vender é **{n_otimo}**, com um valor líquido esperado de **R$ {valor_otimo:.2f}**")
    
    # Mostrar tabela completa
    st.subheader("Tabela Completa de Resultados")
    st.dataframe(df_resultados)

# Tab 3: ROI do Sistema de Informação
with tab3:
    st.header("Análise de ROI para o Sistema de Informação")
    
    st.markdown("""
    Esta análise calcula o Retorno sobre Investimento (ROI) para um sistema de informação
    que promete otimizar a gestão de passagens.
    """)
    
    # Cálculo do ROI
    lucro_investimento = receita_adicional_projetada - custos_operacionais_anuais
    roi = (lucro_investimento / custo_investimento) * 100
    
    # Visualização dos componentes do ROI
    col1, col2, col3 = st.columns(3)
    col1.metric("Receita Adicional", f"R$ {receita_adicional_projetada:.2f}")
    col2.metric("Custos Operacionais", f"R$ {custos_operacionais_anuais:.2f}")
    col3.metric("Custo do Investimento", f"R$ {custo_investimento:.2f}")
    
    # Mostrar resultado do ROI
    st.subheader("Resultado do ROI")
    st.metric("Lucro do Investimento", f"R$ {lucro_investimento:.2f}")
    st.metric("ROI", f"{roi:.2f}%")
    
    # Interpretação do ROI
    st.subheader("Interpretação do ROI")
    if roi > 0:
        st.success(f"O investimento é financeiramente viável com um ROI de {roi:.2f}%")
        
        # Tempo de retorno
        tempo_retorno = custo_investimento / lucro_investimento
        st.info(f"Tempo estimado de retorno do investimento: {tempo_retorno:.2f} anos")
        
        if roi > 50:
            st.markdown("Este é um investimento de **alto retorno**.")
        elif roi > 20:
            st.markdown("Este é um investimento de **bom retorno**.")
        else:
            st.markdown("Este é um investimento de **retorno moderado**.")
    else:
        st.error(f"O investimento não é financeiramente viável com um ROI de {roi:.2f}%")
    
    # Visualização gráfica
    dados_roi = {
        "Componente": ["Receita Adicional", "Custos Operacionais", "Lucro do Investimento"],
        "Valor (R$)": [receita_adicional_projetada, custos_operacionais_anuais, lucro_investimento]
    }
    df_roi = pd.DataFrame(dados_roi)
    
    fig = px.bar(df_roi, x="Componente", y="Valor (R$)", title="Componentes do ROI",
                 color="Componente")
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise de sensibilidade do ROI
    st.subheader("Análise de Sensibilidade do ROI")
    
    # Variação da receita
    variacao_receitas = np.linspace(receita_adicional_projetada * 0.5, receita_adicional_projetada * 1.5, 10)
    roi_por_receita = [(r - custos_operacionais_anuais) / custo_investimento * 100 for r in variacao_receitas]
    
    # Variação dos custos operacionais
    variacao_custos_op = np.linspace(custos_operacionais_anuais * 0.5, custos_operacionais_anuais * 1.5, 10)
    roi_por_custo = [(receita_adicional_projetada - c) / custo_investimento * 100 for c in variacao_custos_op]
    
    # Gráfico para variação de receita
    fig1 = px.line(x=[r/1000 for r in variacao_receitas], y=roi_por_receita, 
                  labels={"x": "Receita Adicional (R$ mil)", "y": "ROI (%)"},
                  title="Sensibilidade do ROI à Variação da Receita Adicional")
    fig1.add_hline(y=0, line_dash="dash", line_color="red")
    fig1.add_vline(x=receita_adicional_projetada/1000, line_dash="dash", line_color="green",
                  annotation_text="Receita Projetada")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico para variação de custos operacionais
    fig2 = px.line(x=[c/1000 for c in variacao_custos_op], y=roi_por_custo, 
                  labels={"x": "Custos Operacionais (R$ mil)", "y": "ROI (%)"},
                  title="Sensibilidade do ROI à Variação dos Custos Operacionais")
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    fig2.add_vline(x=custos_operacionais_anuais/1000, line_dash="dash", line_color="green",
                  annotation_text="Custo Projetado")
    st.plotly_chart(fig2, use_container_width=True)

# Tab 4: Simulações Monte Carlo
with tab4:
    st.header("Simulações Monte Carlo para Análise de Risco do ROI")
    
    st.markdown("""
    Esta análise utiliza simulações Monte Carlo para avaliar o desempenho do sistema em diferentes cenários,
    considerando incertezas no mercado e variações nos parâmetros financeiros.
    """)
    
    # Função para realizar simulações Monte Carlo
    def simular_roi(num_sim, receita_base, custo_op_base, custo_inv, var_receita, var_custos):
        np.random.seed(42)  # Para reprodutibilidade
        
        # Variações para cenários
        var_receita_decimal = var_receita / 100
        var_custos_decimal = var_custos / 100
        
        # Arrays para armazenar resultados
        receitas = np.random.normal(receita_base, receita_base * var_receita_decimal, num_sim)
        custos_op = np.random.normal(custo_op_base, custo_op_base * var_custos_decimal, num_sim)
        
        # Garantir que não haja valores negativos
        receitas = np.maximum(0, receitas)
        custos_op = np.maximum(0, custos_op)
        
        # Calcular lucros e ROIs
        lucros = receitas - custos_op
        rois = (lucros / custo_inv) * 100
        
        return receitas, custos_op, lucros, rois
    
    # Realizar simulações
    receitas_sim, custos_op_sim, lucros_sim, rois_sim = simular_roi(
        num_simulacoes, receita_adicional_projetada, custos_operacionais_anuais,
        custo_investimento, variacao_receita, variacao_custos
    )
    
    # Análise dos resultados da simulação
    roi_medio = np.mean(rois_sim)
    roi_mediano = np.median(rois_sim)
    roi_min = np.min(rois_sim)
    roi_max = np.max(rois_sim)
    prob_roi_positivo = np.mean(rois_sim > 0) * 100
    
    # Mostrar estatísticas das simulações
    st.subheader("Estatísticas das Simulações")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROI Médio", f"{roi_medio:.2f}%")
    col2.metric("ROI Mediano", f"{roi_mediano:.2f}%")
    col3.metric("Probabilidade de ROI Positivo", f"{prob_roi_positivo:.2f}%")
    
    col1, col2 = st.columns(2)
    col1.metric("ROI Mínimo", f"{roi_min:.2f}%")
    col2.metric("ROI Máximo", f"{roi_max:.2f}%")
    
    # Gráfico de distribuição dos ROIs
    fig = px.histogram(rois_sim, nbins=50, title="Distribuição dos ROIs nas Simulações",
                      labels={"value": "ROI (%)", "count": "Frequência"})
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(x=roi_medio, line_dash="dash", line_color="green",
                 annotation_text="ROI Médio")
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de dispersão (receita vs. ROI)
    fig_scatter = px.scatter(x=receitas_sim, y=rois_sim, opacity=0.6,
                            title="Relação entre Receita Adicional e ROI",
                            labels={"x": "Receita Adicional (R$)", "y": "ROI (%)"})
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Definição de cenários
    percentis = np.percentile(rois_sim, [10, 50, 90])
    cenarios = {
        "Pessimista (10%)": percentis[0],
        "Realista (50%)": percentis[1],
        "Otimista (90%)": percentis[2]
    }
    
    # Tabela de cenários
    st.subheader("Análise de Cenários")
    data_cenarios = []
    for cenario, valor in cenarios.items():
        data_cenarios.append({
            "Cenário": cenario,
            "ROI Esperado (%)": valor
        })
    df_cenarios = pd.DataFrame(data_cenarios)
    st.dataframe(df_cenarios)
    
    # Gráfico de cenários
    fig_cenarios = go.Figure()
    cores = ["red", "blue", "green"]
    
    for i, (cenario, valor) in enumerate(cenarios.items()):
        fig_cenarios.add_trace(go.Bar(
            x=[cenario],
            y=[valor],
            name=cenario,
            marker_color=cores[i]
        ))
    
    fig_cenarios.update_layout(
        title="ROI Esperado por Cenário",
        yaxis_title="ROI (%)",
        showlegend=False
    )
    
    st.plotly_chart(fig_cenarios, use_container_width=True)
    
    # Análise de valor em risco (VaR)
    var_95 = np.percentile(rois_sim, 5)
    st.subheader("Análise de Valor em Risco (VaR)")
    st.markdown(f"""
    O **Valor em Risco (VaR)** com 95% de confiança é de **{abs(min(0, var_95)):.2f}%** de perda no ROI.
    
    Isso significa que, com 95% de confiança, a perda máxima no ROI não ultrapassará {abs(min(0, var_95)):.2f}%.
    """)
    
    # Conclusão geral
    st.subheader("Conclusão da Análise de Risco")
    if prob_roi_positivo > 80:
        st.success(f"""
        O investimento no sistema de informação apresenta **baixo risco**, com {prob_roi_positivo:.2f}% de probabilidade 
        de gerar um ROI positivo. A análise Monte Carlo sugere um ROI médio esperado de {roi_medio:.2f}%.
        """)
    elif prob_roi_positivo > 50:
        st.warning(f"""
        O investimento no sistema de informação apresenta **risco moderado**, com {prob_roi_positivo:.2f}% de probabilidade 
        de gerar um ROI positivo. A análise Monte Carlo sugere um ROI médio esperado de {roi_medio:.2f}%.
        """)
    else:
        st.error(f"""
        O investimento no sistema de informação apresenta **alto risco**, com apenas {prob_roi_positivo:.2f}% de probabilidade 
        de gerar um ROI positivo. A análise Monte Carlo sugere um ROI médio esperado de {roi_medio:.2f}%.
        """)

# Conclusão geral do aplicativo
st.markdown("---")
st.header("Conclusões e Recomendações")

# Obter conclusões de cada análise
if "max_passagens_seguras" in locals() and max_passagens_seguras > 0:
    conclusao_overbooking = f"Vender até {max_passagens_seguras} passagens é seguro mantendo o risco abaixo de {risco_maximo*100}%."
else:
    conclusao_overbooking = "Não foi possível identificar um número seguro de passagens extras com os parâmetros atuais."

if "n_otimo" in locals():
    conclusao_viabilidade = f"O número ótimo de passagens extras a vender é {n_otimo}, com valor líquido esperado de R$ {valor_otimo:.2f}."
else:
    conclusao_viabilidade = "A análise de viabilidade não pôde determinar um número ótimo de passagens extras."

if "roi" in locals():
    if roi > 0:
        conclusao_roi = f"O investimento no sistema de informação é viável com ROI de {roi:.2f}%."
    else:
        conclusao_roi = f"O investimento no sistema de informação não é viável com ROI de {roi:.2f}%."
else:
    conclusao_roi = "Não foi possível calcular o ROI com os parâmetros fornecidos."

if "prob_roi_positivo" in locals():
    conclusao_simulacao = f"As simulações indicam {prob_roi_positivo:.2f}% de chance de ROI positivo, com ROI médio de {roi_medio:.2f}%."
else:
    conclusao_simulacao = "Não foi possível realizar as simulações com os parâmetros fornecidos."

# Exibir conclusões
st.markdown(f"""
### Resumo das Conclusões:

1. **Análise de Overbooking:** {conclusao_overbooking}

2. **Viabilidade Financeira:** {conclusao_viabilidade}

3. **ROI do Sistema:** {conclusao_roi}

4. **Análise de Risco:** {conclusao_simulacao}

### Recomendações Finais:

Com base nas análises realizadas, recomendamos:

- Implementar uma política de overbooking controlado, mantendo o número de passagens vendidas dentro do limite seguro calculado.
- Investir no sistema de informação proposto, que demonstra viabilidade financeira e bom potencial de retorno.
- Monitorar constantemente os parâmetros reais (taxa de comparecimento, custos de compensação) para ajustar as estratégias.
- Considerar a implementação de um sistema de incentivos para passageiros voluntariamente cederem seus lugares em casos de overbooking.

Este aplicativo pode ser utilizado para simular diferentes cenários e ajustar parâmetros conforme as condições de mercado se alterem.
""")

# Informações finais e disclaimer
st.markdown("---")
st.caption("""
**Disclaimer:** Este aplicativo foi desenvolvido para fins de demonstração. Os dados e análises são fictícios e 
não devem ser utilizados para tomada de decisões reais sem validação adicional.

© 2025 Análise de Operações Aéreas - Desenvolvido para Aérea Confiável
""")
