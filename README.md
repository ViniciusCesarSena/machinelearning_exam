Show! Montei um **README.md** prontinho pra esse repositório — direto ao ponto, bonitinho e com instruções para rodar localmente (Jupyter e Streamlit), estrutura do projeto e notas de troubleshooting.

> Copie e cole o conteúdo abaixo em `README.md` na raiz do repo.

---

# Machine Learning Exam — Walmart (Módulos 1 e 2)

**Prova de Machine Learning** com foco em **EDA**, **engenharia de atributos**, **PCA** e **clusterização hierárquica** aplicada a uma base de transações do Walmart. Inclui notebooks, app Streamlit para visualização e relatórios em PDF/LaTeX. ([GitHub][1])

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)]()

</div>

---

## 📁 Estrutura do projeto

```
machinelearning_exam/
├─ notebooks/           # Jupyter Notebooks (EDA, PCA, Clusterização, etc.)
├─ reports/             # Relatórios e artefatos (PDF/TeX, imagens)
├─ streamlit_app.py     # App Streamlit para exploração interativa
├─ requirements.txt     # Dependências de execução
└─ walmart.csv          # Base de dados (Walmart) usada nos notebooks/app
```

> **Observação:** a presença de `notebooks/`, `reports/`, `requirements.txt`, `streamlit_app.py` e `walmart.csv` vem do próprio repositório. ([GitHub][1])

---

## 🚀 Como rodar localmente

### 1) Clonar e criar ambiente

```bash
git clone https://github.com/ViniciusCesarSena/machinelearning_exam.git
cd machinelearning_exam

# opcional, mas recomendado
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2) Instalar dependências

```bash
pip install -r requirements.txt
```

> Se aparecer erro de compilação LaTeX ao gerar PDFs em `reports/`, instale uma distribuição LaTeX (ex.: TeX Live completo no Linux/macOS, MikTeX no Windows).

### 3) Executar os notebooks

```bash
jupyter notebook
```

Abra os arquivos em `notebooks/` e rode as células em ordem. Os notebooks cobrem:

* **EDA**: estatísticas descritivas, distribuição, outliers, cauda longa
* **Pré-processamento**: imputação e codificação
* **Engenharia de atributos**: métricas por usuário (volume, intensidade, variedade)
* **PCA**: padronização e redução de dimensionalidade
* **Clusterização**: método aglomerativo de Ward + silhueta

### 4) Rodar o app Streamlit (opcional)

```bash
streamlit run streamlit_app.py
```

O app carrega `walmart.csv` e permite explorar interativamente métricas, PCA e clusters.

---

## 🧠 Metodologia (resumo)

1. **EDA**: ticket médio, mediana, assimetria, categorias mais frequentes, estratificação por cidade/tipo.
2. **Pré-processamento**: tratamento de nulos, codificações e conversões numéricas.
3. **Engenharia de atributos**: agregações por usuário (gasto acumulado, ticket médio, desvio, nº de produtos/categorias únicas, participação nas top categorias).
4. **Normalização + PCA**: `z-score`; seleção de componentes com variância acumulada alvo.
5. **Clusterização hierárquica (Ward)**: avaliação de (k \in [2,6]) com **silhueta**; seleção do melhor (k) para macrosegmentação.

---

## 📊 Resultados esperados

* **2 macroperfis** bem distintos em volume/frequência/variedade.
* Tabelas e figuras com métricas por cluster (usuários %, compras/ano, total gasto médio, ticket médio, etc.).
* Relatório final em `reports/` com interpretação de negócio e recomendações.

> Os números exatos variam conforme parametrizações e filtros dos notebooks, mas a estrutura e as saídas (tabelas/figuras) seguem esse fluxo.

---

## 🔧 Troubleshooting

* **`ModuleNotFoundError: matplotlib.pyplot` no Streamlit**
  Certifique-se de que instalou as dependências no **mesmo ambiente** em que executa o `streamlit`:

  ```bash
  which python
  which streamlit
  pip show matplotlib
  ```

  Se `streamlit` estiver fora do venv, reinstale dentro do venv: `pip install streamlit matplotlib`.

* **Problemas com acentuação (pt-BR) em LaTeX**
  Garanta no preâmbulo:
  `\usepackage[utf8]{inputenc}`, `\usepackage[T1]{fontenc}`, `\usepackage[portuguese]{babel}`.

* **Alinhamento de números e R$ em tabelas**
  Use `siunitx` (`S` columns) e deixe `R$` **apenas no cabeçalho** para manter o alinhamento por decimal.

---

## 🗂️ Dados

O arquivo `walmart.csv` na raiz é a base utilizada para EDA, PCA e clusterização. Se preferir rodar com outra base, ajuste os caminhos nos notebooks e em `streamlit_app.py`.


---

## 🙋‍♂️ Autor

* **Vinícius César Sena Torres** — módulos 1 e 2 de Machine Learning (prova).
* Repositório no GitHub: `ViniciusCesarSena/machinelearning_exam`. ([GitHub][1])

---

[1]: https://github.com/ViniciusCesarSena/machinelearning_exam "GitHub - ViniciusCesarSena/machinelearning_exam: Prova de Machine Learning, resolução dos módulos 1 e 2"
