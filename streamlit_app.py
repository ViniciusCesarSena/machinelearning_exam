from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "walmart.csv"

WALMART_BLUE = "#0071ce"
WALMART_YELLOW = "#ffc220"
WALMART_SLATE = "#004690"


def inject_theme() -> None:
    st.set_page_config(
        page_title="Walmart Analytics Hub",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    custom_css = f"""
    <style>
        :root {{
            --walmart-blue: {WALMART_BLUE};
            --walmart-yellow: {WALMART_YELLOW};
            --walmart-slate: {WALMART_SLATE};
        }}
        .block-container {{
            padding-top: 1.5rem;
        }}
        .walmart-card {{
            background: #ffffff;
            border-left: 4px solid var(--walmart-blue);
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }}
        .stMetric label {{
            color: var(--walmart-slate);
            font-weight: 600;
        }}
        .analysis-note {{
            border-left: 4px solid var(--walmart-yellow);
            padding: 0.75rem 1rem;
            background: #fffdf4;
            border-radius: 0.5rem;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }}
        .code-tag {{
            background: var(--walmart-blue);
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_data(show_spinner="Carregando base do Walmart...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data(show_spinner="Calculando indicadores do dashboard...")
def compute_dashboard_artifacts(df: pd.DataFrame) -> Dict[str, object]:
    purchase_desc = df["Purchase"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    top_categories = (
        df["Product_Category"].fillna(-1).value_counts().head(10).rename("volume")
    )

    gender_mix = (
        df.groupby(["Gender", "Age"])["Purchase"]
        .mean()
        .rename("avg_purchase")
        .reset_index()
    )

    city_performance = (
        df.assign(
            Stay_Num=df["Stay_In_Current_City_Years"]
            .str.replace("+", "", regex=False)
            .astype(int)
        )
        .groupby(["City_Category", "Stay_Num"])["Purchase"]
        .mean()
        .rename("avg_purchase")
        .reset_index()
    )

    age_order = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
    gender_age_pivot = (
        df.groupby(["Gender", "Age"])["Purchase"]
        .mean()
        .unstack("Age")
        .reindex(columns=age_order, fill_value=np.nan)
    )

    return {
        "purchase_desc": purchase_desc,
        "top_categories": top_categories,
        "gender_mix": gender_mix,
        "city_performance": city_performance,
        "gender_age_pivot": gender_age_pivot,
    }


@st.cache_data(show_spinner="Preparando features de clusterização...")
def prepare_cluster_artifacts(df: pd.DataFrame) -> Dict[str, object]:
    age_order = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
    age_map = {age: idx for idx, age in enumerate(age_order)}

    work_df = df.copy()
    work_df["Product_Category"] = work_df["Product_Category"].fillna(-1)
    work_df["Stay_In_Current_City_Years"] = (
        work_df["Stay_In_Current_City_Years"].str.replace("+", "", regex=False).astype(int)
    )
    work_df["Gender_Code"] = work_df["Gender"].map({"F": 0, "M": 1})
    work_df["Age_Code"] = work_df["Age"].map(age_map)

    user_base = (
        work_df.groupby("User_ID")
        .agg(
            total_trans=("Purchase", "count"),
            total_spent=("Purchase", "sum"),
            mean_spent=("Purchase", "mean"),
            median_spent=("Purchase", "median"),
            std_spent=("Purchase", "std"),
            unique_products=("Product_ID", "nunique"),
            unique_categories=("Product_Category", "nunique"),
            gender=("Gender_Code", "first"),
            age_code=("Age_Code", "first"),
            occupation=("Occupation", "first"),
            city=("City_Category", "first"),
            stay_years=("Stay_In_Current_City_Years", "first"),
            marital_status=("Marital_Status", "first"),
        )
        .reset_index()
    )

    user_base["std_spent"] = user_base["std_spent"].fillna(0)
    user_base["freq_por_ano"] = user_base["total_trans"] / np.where(
        user_base["stay_years"] == 0, 1, user_base["stay_years"]
    )

    top_categories = work_df["Product_Category"].value_counts().head(5).index
    category_share = (
        work_df[work_df["Product_Category"].isin(top_categories)]
        .groupby(["User_ID", "Product_Category"])["Purchase"]
        .count()
        .unstack(fill_value=0)
    )

    category_share = category_share.div(category_share.sum(axis=1), axis=0).fillna(0)
    category_share = category_share.add_prefix("share_cat_")

    feature_df = user_base.merge(category_share, on="User_ID", how="left").fillna(0)
    feature_df = pd.concat(
        [feature_df, pd.get_dummies(feature_df["city"], prefix="city")], axis=1
    ).drop(columns=["city"])
    feature_df = pd.concat(
        [feature_df, pd.get_dummies(feature_df["marital_status"], prefix="marital")],
        axis=1,
    ).drop(columns=["marital_status"])

    feature_cols = [col for col in feature_df.columns if col != "User_ID"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df[feature_cols])

    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X_scaled)
    viz_pca = PCA(n_components=3)
    X_pca_viz = viz_pca.fit_transform(X_scaled)

    rng = np.random.default_rng(42)
    sample_size = min(400, X_scaled.shape[0])
    sample_idx = rng.choice(X_scaled.shape[0], size=sample_size, replace=False)
    linkage_matrix = linkage(X_scaled[sample_idx], method="ward", metric="euclidean")

    silhouette_results = []
    for k in range(2, 7):
        model = AgglomerativeClustering(
            n_clusters=k, linkage="ward", metric="euclidean"
        )
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels, metric="euclidean")
        silhouette_results.append({"k": k, "silhouette": score})
    silhouette_df = pd.DataFrame(silhouette_results)
    best_k = int(silhouette_df.loc[silhouette_df["silhouette"].idxmax(), "k"])

    final_model = AgglomerativeClustering(
        n_clusters=best_k, linkage="ward", metric="euclidean"
    )
    clusters = final_model.fit_predict(X_scaled)

    feature_df = feature_df.copy()
    feature_df["cluster"] = clusters
    feature_df["pca1"] = X_pca_viz[:, 0]
    feature_df["pca2"] = X_pca_viz[:, 1]

    cluster_sizes = feature_df["cluster"].value_counts().rename("qtd_usuarios")
    cluster_summary = (
        feature_df.groupby("cluster")[
            [
                "total_trans",
                "total_spent",
                "mean_spent",
                "median_spent",
                "std_spent",
                "unique_products",
                "unique_categories",
                "freq_por_ano",
            ]
        ]
        .mean()
        .round(2)
    )
    cat_cols = [col for col in feature_df.columns if col.startswith("share_cat_")]
    category_summary = feature_df.groupby("cluster")[cat_cols].mean().round(3)
    demographics = (
        feature_df.groupby("cluster")[["gender", "age_code", "occupation", "stay_years"]]
        .mean()
        .rename(
            columns={
                "gender": "proporcao_homens",
                "age_code": "idade_media_cod",
                "occupation": "ocupacao_media",
                "stay_years": "tempo_medio_cidade",
            }
        )
        .round(2)
    )

    total_users = cluster_sizes.sum()
    cluster_narratives = [
        (
            f"Cluster {cluster_id}: {cluster_sizes.loc[cluster_id] / total_users:.1%} "
            f"dos usuários, {row['total_trans']:.1f} compras/cliente, ticket médio "
            f"de R${row['mean_spent']:.2f} e {row['unique_categories']:.1f} categorias únicas."
        )
        for cluster_id, row in cluster_summary.iterrows()
    ]

    return {
        "feature_df": feature_df,
        "linkage_matrix": linkage_matrix,
        "silhouette_df": silhouette_df,
        "cluster_summary": cluster_summary,
        "cluster_sizes": cluster_sizes,
        "category_summary": category_summary,
        "demographics": demographics,
        "cluster_narratives": cluster_narratives,
    }


@st.cache_data(show_spinner="Calculando regras associativas...")
def prepare_market_basket(df: pd.DataFrame) -> Dict[str, object]:
    product_freq = df["Product_ID"].value_counts()
    top_products = product_freq.head(50)
    filtered = df[df["Product_ID"].isin(top_products.index)].copy()

    transaction_matrix = (
        filtered.groupby(["User_ID", "Product_ID"])
        .size()
        .unstack(fill_value=0)
    )
    transaction_bool = transaction_matrix > 0
    transaction_counts = transaction_bool.astype(int)

    frequent_itemsets = apriori(
        transaction_bool, min_support=0.03, use_colnames=True
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].str.len()

    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=1.05
    )
    rules = rules[rules["confidence"] >= 0.3].sort_values(by="lift", ascending=False)

    def format_itemset(itemset: frozenset) -> str:
        return ", ".join(sorted(itemset))

    rules = rules.assign(
        antecedentes=rules["antecedents"].apply(format_itemset),
        consequentes=rules["consequents"].apply(format_itemset),
    )

    top_for_heatmap = top_products.index[:12]
    co_matrix = transaction_counts[top_for_heatmap].T.dot(
        transaction_counts[top_for_heatmap]
    )
    co_support = co_matrix / transaction_counts.shape[0]

    resumo_regras = [
        (
            f"{row['antecedentes']} => {row['consequentes']} | suporte {row['support']:.3f}, "
            f"confiança {row['confidence']:.2f}, lift {row['lift']:.2f}"
        )
        for _, row in rules.head(5).iterrows()
    ]

    return {
        "product_freq": product_freq,
        "transaction_bool": transaction_bool,
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "co_support": co_support,
        "resumo_regras": resumo_regras,
    }


def fig_purchase_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        df["Purchase"],
        bins=40,
        kde=True,
        color=WALMART_BLUE,
        ax=ax,
    )
    ax.set_title("Distribuição do valor de compra (Ticket)")
    ax.set_xlabel("Valor em R$")
    ax.set_ylabel("Frequência")
    return fig


def fig_gender_age_heatmap(pivot_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Ticket médio por gênero e faixa etária")
    ax.set_xlabel("Faixa etária")
    ax.set_ylabel("Gênero")
    return fig


def fig_top_categories(top_categories: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=top_categories.values,
        y=top_categories.index.astype(str),
        palette=[WALMART_YELLOW] + [WALMART_BLUE] * (len(top_categories) - 1),
        ax=ax,
    )
    ax.set_title("Top 10 categorias de produto")
    ax.set_xlabel("Volume de transações")
    ax.set_ylabel("Categoria")
    return fig


def fig_city_performance(city_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        data=city_df,
        x="Stay_Num",
        y="avg_purchase",
        hue="City_Category",
        marker="o",
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Ticket médio x Tempo na cidade")
    ax.set_xlabel("Anos na cidade")
    ax.set_ylabel("Ticket médio (R$)")
    return fig


def fig_dendrogram(linkage_matrix: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(
        linkage_matrix,
        truncate_mode="level",
        p=5,
        ax=ax,
        color_threshold=None,
    )
    ax.set_title("Dendrograma (amostra de usuários)")
    ax.set_xlabel("Observações agrupadas")
    ax.set_ylabel("Distância (Ward)")
    return fig


def fig_silhouette(silhouette_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(
        data=silhouette_df,
        x="k",
        y="silhouette",
        marker="o",
        color=WALMART_BLUE,
        ax=ax,
    )
    ax.set_title("Índice de silhueta por quantidade de clusters")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhueta média")
    return fig


def fig_cluster_scatter(feature_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=feature_df,
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="Set2",
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Clusters projetados em PCA (2D)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(title="Cluster")
    return fig


def fig_top_rules_scatter(rules: pd.DataFrame) -> plt.Figure:
    top_rules = rules.head(30)
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = sns.scatterplot(
        data=top_rules,
        x="support",
        y="confidence",
        size="lift",
        hue="lift",
        palette="viridis",
        sizes=(50, 400),
        ax=ax,
    )
    ax.set_title("Regras: suporte x confiança (tamanho = lift)")
    ax.set_xlabel("Suporte")
    ax.set_ylabel("Confiança")
    handles, labels = scatter.get_legend_handles_labels()
    if handles:
        ax.legend(handles[1:], labels[1:], title="Lift", loc="best")
    return fig


def fig_cooccurrence_heatmap(co_support: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(co_support, cmap="Reds", annot=False, ax=ax)
    ax.set_title("Suporte conjunto entre os principais produtos")
    ax.set_xlabel("Produto")
    ax.set_ylabel("Produto")
    return fig


def code_block(fn) -> None:
    st.caption("CÓDIGO")
    st.code(inspect.getsource(fn), language="python")


def dashboard_tab(df: pd.DataFrame) -> None:
    st.subheader("Dashboard Estratégico")
    artifacts = compute_dashboard_artifacts(df)
    purchase_desc = artifacts["purchase_desc"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticket médio", f"R${purchase_desc['mean']:,.0f}")
    col2.metric("Mediana", f"R${purchase_desc['50%']:,.0f}")
    col3.metric("Usuários únicos", f"{df['User_ID'].nunique():,}")
    col4.metric("Produtos distintos", f"{df['Product_ID'].nunique():,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(fig_purchase_distribution(df), use_container_width=True)
        st.markdown(
            "<div class='analysis-note'>A distribuição confirma cauda longa com compras "
            "acima de R$15 mil puxando a média; reforça uso de métricas robustas na "
            "clusterização.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver código"):
            code_block(fig_purchase_distribution)
    with col_b:
        st.pyplot(
            fig_gender_age_heatmap(artifacts["gender_age_pivot"]),
            use_container_width=True,
        )
        st.markdown(
            "<div class='analysis-note'>Mulheres até 35 anos apresentam ticket mais alto "
            "que homens na mesma faixa, mas o padrão se inverte após os 40 anos, "
            "sugerindo campanhas segmentadas por idade.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver código"):
            code_block(fig_gender_age_heatmap)

    col_c, col_d = st.columns(2)
    with col_c:
        st.pyplot(fig_top_categories(artifacts["top_categories"]), use_container_width=True)
        st.markdown(
            "<div class='analysis-note'>As três categorias líderes concentram mais de "
            "70% das compras, justificando o foco nas variáveis comportamentais usadas "
            "no módulo de clusters.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver código"):
            code_block(fig_top_categories)
    with col_d:
        st.pyplot(fig_city_performance(artifacts["city_performance"]), use_container_width=True)
        st.markdown(
            "<div class='analysis-note'>Cidades tipo C sustentam o ticket mais alto "
            "independentemente do tempo de residência, o que valida tratar localização "
            "como atributo-chave nas recomendações.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver código"):
            code_block(fig_city_performance)


def module1_tab(df: pd.DataFrame) -> None:
    st.subheader("Módulo 1 · Clusterização hierárquica")
    artifacts = prepare_cluster_artifacts(df)

    st.markdown(
        "<div class='walmart-card'>"
        "<span class='code-tag'>PIPELINE</span> "
        "Agrupamos usuários por intensidade, valor e variedade de compras após PCA e "
        "clusterização de Ward. Abaixo estão todas as etapas executadas.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Ver código da preparação e clusterização", expanded=False):
        code_block(prepare_cluster_artifacts)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_dendrogram(artifacts["linkage_matrix"]), use_container_width=True)
        st.markdown(
            "<div class='analysis-note'>O dendrograma evidencia dois níveis claros de "
            "separação: usuários de altíssima recorrência destacam-se cedo na árvore.</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.pyplot(fig_silhouette(artifacts["silhouette_df"]), use_container_width=True)
        best_k = int(artifacts["silhouette_df"].loc[
            artifacts["silhouette_df"]["silhouette"].idxmax(), "k"
        ])
        st.markdown(
            f"<div class='analysis-note'>A silhueta máxima ocorre em k={best_k}, "
            "equilibrando compacidade e separação para macrosegmentação.</div>",
            unsafe_allow_html=True,
        )

    st.pyplot(fig_cluster_scatter(artifacts["feature_df"]), use_container_width=True)
    st.markdown(
        "<div class='analysis-note'>Mesmo após a redução PCA, os clusters mostram "
        "fronteiras bem definidas: o grupo C1 concentra usuários intensivos.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Tabelas resumidas**")
    st.dataframe(artifacts["cluster_sizes"].to_frame(), use_container_width=True)
    st.dataframe(artifacts["cluster_summary"], use_container_width=True)
    st.dataframe(artifacts["category_summary"], use_container_width=True)
    st.dataframe(artifacts["demographics"], use_container_width=True)

    st.markdown("**Interpretações**")
    for note in artifacts["cluster_narratives"]:
        st.markdown(f"- {note}")


def module2_tab(df: pd.DataFrame) -> None:
    st.subheader("Módulo 2 · Regras associativas")
    artifacts = prepare_market_basket(df)

    with st.expander("Ver código das regras de associação"):
        code_block(prepare_market_basket)

    col1, col2 = st.columns([1, 1])
    with col1:
        top15 = artifacts["product_freq"].head(15)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=top15.values,
            y=top15.index,
            color=WALMART_BLUE,
            ax=ax,
        )
        ax.set_title("Top 15 produtos mais vendidos")
        ax.set_xlabel("Registros")
        ax.set_ylabel("Produto")
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            "<div class='analysis-note'>O volume concentrado nos primeiros SKUs valida "
            "o corte em 50 itens para manter suporte acima de 3%.</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.pyplot(fig_top_rules_scatter(artifacts["rules"]), use_container_width=True)
        st.markdown(
            "<div class='analysis-note'>As regras com maior lift combinam itens das "
            "categorias 5 e 1, sinalizando combos naturais para kits promocionais.</div>",
            unsafe_allow_html=True,
        )

    st.pyplot(fig_cooccurrence_heatmap(artifacts["co_support"]), use_container_width=True)
    st.markdown(
        "<div class='analysis-note'>O mapa de calor revela blocos diagonais intensos "
        "que representam famílias de produtos compradas juntas; ideal para "
        "recomendações automatizadas.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Top 10 regras formatadas**")
    rules_view = artifacts["rules"][
        ["antecedentes", "consequentes", "support", "confidence", "lift"]
    ].head(10)
    st.dataframe(rules_view, use_container_width=True)

    st.markdown("**Conclusões**")
    for resumo in artifacts["resumo_regras"]:
        st.markdown(f"- {resumo}")


def main() -> None:
    inject_theme()
    df = load_data()

    st.title("Walmart Analytics Hub")
    st.caption(
        "Relatórios interativos criados com Streamlit para apresentar os achados dos "
        "módulos de Clusterização e Regras Associativas."
    )

    st.sidebar.header("Configurações")
    st.sidebar.write("Fonte de dados: `walmart.csv`")
    st.sidebar.write("Cores oficiais do Walmart aplicadas automaticamente.")
    st.sidebar.info(
        "Use as abas abaixo para navegar pelo dashboard, explorar os clusters e "
        "consultar as regras de mercado."
    )

    tabs = st.tabs(
        [
            "Dashboard",
            "Módulo 1 - Clusterização",
            "Módulo 2 - Regras associativas",
        ]
    )

    with tabs[0]:
        dashboard_tab(df)
    with tabs[1]:
        module1_tab(df)
    with tabs[2]:
        module2_tab(df)


if __name__ == "__main__":
    main()
