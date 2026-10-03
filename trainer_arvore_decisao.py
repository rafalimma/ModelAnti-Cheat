"""Treina uma arvore de decisao explicavel usando CSVs de janelas.

O script evita dois vazamentos comuns em telemetria de anti-cheat:

1. Nao usa pontuacoes, alertas ou decisoes produzidas pelo motor de regras.
2. Nunca separa aleatoriamente janelas da mesma partida entre treino e teste.

Os rotulos devem vir de um manifesto independente, preenchido a partir do
roteiro do experimento controlado. Exemplo de manifesto CSV:

arquivo_csv,partida_id,jogador_sessao,inicio_s,fim_s,rotulo_real,jogador_global
datasets/partida_normal_raw_janelas.csv,normal_01,1,0,900,normal,pessoa_a
datasets/teste_pequeno_janelas.csv,hacker_01,2,262,290,esp,pessoa_b

Cada linha do manifesto rotula as janelas cujo ponto central esta dentro do
intervalo [inicio_s, fim_s]. Intervalos podem ser usados para nao marcar como
hacker os trechos normais de uma partida que contem apenas uma acao hacker.

Uso:
    python trainer_arvore_decisao.py --manifesto datasets/rotulos_treino.csv

Para criar somente o cabecalho do manifesto:
    python trainer_arvore_decisao.py --criar-template datasets/rotulos_treino.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Evita depender de permissao de escrita no perfil do usuario para o cache de fontes.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "model_anticheat_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# Lista branca: medidas comportamentais genericas. Foram excluidas features que
# ja embutem a regra de 300 m, contagens de episodios e todas as decisoes finais.
FEATURES_COMPORTAMENTAIS = [
    # Movimento / speedhack
    "vel_posicao_media",
    "vel_posicao_max",
    "aceleracao_media",
    "aceleracao_max",
    # Movimento de camera / aimbot
    "vel_rotacao_media",
    "vel_rotacao_max",
    "vel_rotacao_std",
    "jitter_medio",
    "entropia_media",
    "eficiencia_trajeto_media",
    "variacao_camera_media",
    "angulo_alvo_medio",
    "angulo_alvo_min",
    # Contexto do alvo / ESP
    "distancia_alvo_media",
    "distancia_alvo_max",
    "taxa_alvo_encontrado",
    "taxa_alvo_na_mira",
    "taxa_visao_direta",
    "taxa_alvo_oculto",
    "taxa_mirando_alvo",
    "taxa_mira_oculta",
    "duracao_rastreando_oculto_max",
    "duracao_mirando_oculto_max",
]

COLUNAS_PROIBIDAS = {
    "jogador_sessao",
    "janela_analise_10s",
    "inicio_janela_s",
    "fim_janela_s",
    "pontos_evento_total",
    "pontos_suspeita_60s_max",
    "tipos_evidencia_60s_max",
    "nivel_suspeita_max",
    "indice_regra_0a1",
    "padrao_suspeito",
    "candidato_revisao",
    "candidato_banimento",
}

COLUNAS_MANIFESTO = [
    "arquivo_csv",
    "partida_id",
    "jogador_sessao",
    "inicio_s",
    "fim_s",
    "rotulo_real",
    "jogador_global",
]

ROTULOS_NORMAIS = {"0", "normal", "legitimo", "legitima", "limpo", "limpa"}
ROTULOS_HACKER = {"1", "hacker", "esp", "wallhack", "aimbot", "speedhack"}


def normalizar_rotulo(valor: object) -> tuple[int, str]:
    texto = str(valor).strip().lower()
    if texto in ROTULOS_NORMAIS:
        return 0, "normal"
    if texto in ROTULOS_HACKER:
        tipo = "hacker" if texto == "1" else texto
        return 1, tipo
    raise ValueError(
        f"Rotulo desconhecido: {valor!r}. Use normal, hacker, esp, wallhack, "
        "aimbot ou speedhack."
    )


def criar_template(destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        raise FileExistsError(f"O arquivo ja existe e nao sera sobrescrito: {destino}")
    pd.DataFrame(columns=COLUNAS_MANIFESTO).to_csv(destino, index=False)
    print(f"Template criado em: {destino}")


def resolver_arquivo(caminho: str, base_manifesto: Path) -> Path:
    arquivo = Path(caminho)
    if arquivo.is_absolute():
        return arquivo

    candidato_manifesto = (base_manifesto / arquivo).resolve()
    if candidato_manifesto.exists():
        return candidato_manifesto

    return (Path.cwd() / arquivo).resolve()


def carregar_manifesto(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Manifesto nao encontrado: {caminho}")

    manifesto = pd.read_csv(caminho)
    obrigatorias = set(COLUNAS_MANIFESTO[:-1])
    faltantes = sorted(obrigatorias - set(manifesto.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes no manifesto: {faltantes}")
    if manifesto.empty:
        raise ValueError("O manifesto esta vazio. Rotule os intervalos antes de treinar.")

    manifesto = manifesto.copy()
    manifesto["jogador_sessao"] = pd.to_numeric(
        manifesto["jogador_sessao"], errors="raise"
    ).astype(int)
    manifesto["inicio_s"] = pd.to_numeric(manifesto["inicio_s"], errors="raise")
    manifesto["fim_s"] = pd.to_numeric(manifesto["fim_s"], errors="raise")
    if (manifesto["fim_s"] < manifesto["inicio_s"]).any():
        raise ValueError("Ha intervalo com fim_s menor que inicio_s no manifesto.")

    normalizados = manifesto["rotulo_real"].map(normalizar_rotulo)
    manifesto["classe_binaria"] = normalizados.map(lambda item: item[0])
    manifesto["tipo_rotulo"] = normalizados.map(lambda item: item[1])
    if "jogador_global" not in manifesto:
        manifesto["jogador_global"] = ""
    manifesto["jogador_global"] = manifesto["jogador_global"].fillna("").astype(str)
    return manifesto


def carregar_intervalos_rotulados(manifesto: pd.DataFrame, base_manifesto: Path) -> pd.DataFrame:
    partes: list[pd.DataFrame] = []
    cache: dict[Path, pd.DataFrame] = {}

    for ordem, linha in manifesto.reset_index(drop=True).iterrows():
        arquivo = resolver_arquivo(str(linha["arquivo_csv"]), base_manifesto)
        if not arquivo.exists():
            raise FileNotFoundError(f"CSV de janelas nao encontrado: {arquivo}")

        if arquivo not in cache:
            df = pd.read_csv(arquivo)
            requeridas = {
                "jogador_sessao",
                "janela_analise_10s",
                "inicio_janela_s",
                "fim_janela_s",
            }
            ausentes = sorted(requeridas - set(df.columns))
            if ausentes:
                raise ValueError(f"{arquivo.name}: colunas obrigatorias ausentes: {ausentes}")
            cache[arquivo] = df

        df = cache[arquivo]
        centro = (df["inicio_janela_s"] + df["fim_janela_s"]) / 2.0
        mascara = (
            (df["jogador_sessao"].astype(int) == int(linha["jogador_sessao"]))
            & (centro >= float(linha["inicio_s"]))
            & (centro <= float(linha["fim_s"]))
        )
        trecho = df.loc[mascara].copy()
        if trecho.empty:
            print(
                "AVISO: nenhum registro encontrado para "
                f"{arquivo.name}, jogador {linha['jogador_sessao']}, "
                f"intervalo {linha['inicio_s']}-{linha['fim_s']} s."
            )
            continue

        trecho["arquivo_origem"] = str(arquivo)
        trecho["partida_id"] = str(linha["partida_id"])
        trecho["jogador_global"] = str(linha["jogador_global"]).strip()
        trecho["rotulo_real"] = int(linha["classe_binaria"])
        trecho["tipo_rotulo"] = str(linha["tipo_rotulo"])
        trecho["ordem_manifesto"] = ordem
        partes.append(trecho)

    if not partes:
        raise ValueError("Nenhuma janela foi rotulada pelos intervalos do manifesto.")

    dados = pd.concat(partes, ignore_index=True)
    chave = ["arquivo_origem", "jogador_sessao", "janela_analise_10s"]
    conflitos = dados.groupby(chave, dropna=False)["rotulo_real"].nunique()
    conflitos = conflitos[conflitos > 1]
    if not conflitos.empty:
        exemplo = conflitos.index[0]
        raise ValueError(
            "O manifesto atribui rotulos conflitantes a uma mesma janela. "
            f"Primeiro conflito: {exemplo}"
        )

    # Sobreposicoes com o mesmo rotulo nao devem duplicar amostras.
    dados = dados.sort_values("ordem_manifesto").drop_duplicates(chave, keep="last")
    return dados.reset_index(drop=True)


def selecionar_features(dados: pd.DataFrame) -> list[str]:
    features = [coluna for coluna in FEATURES_COMPORTAMENTAIS if coluna in dados.columns]
    faltantes = [coluna for coluna in FEATURES_COMPORTAMENTAIS if coluna not in dados.columns]
    if faltantes:
        print(f"AVISO: features ausentes e ignoradas: {faltantes}")
    if len(features) < 10:
        raise ValueError(
            f"Somente {len(features)} features validas foram encontradas; esperado >= 10."
        )
    vazamento = sorted(set(features) & COLUNAS_PROIBIDAS)
    if vazamento:
        raise AssertionError(f"Features proibidas entraram no modelo: {vazamento}")
    return features


def escolher_split_por_grupo(
    y: pd.Series,
    grupos: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    grupos_unicos = pd.Series(grupos).nunique()
    if grupos_unicos < 3:
        raise ValueError(
            "Sao necessarias pelo menos 3 unidades independentes no grupo de validacao. "
            f"Foram encontradas {grupos_unicos}. Colete/rotule mais partidas ou jogadores."
        )

    melhor: tuple[float, np.ndarray, np.ndarray] | None = None
    gerador = GroupShuffleSplit(
        n_splits=300,
        test_size=test_size,
        random_state=random_state,
    )
    proporcao_global = float(y.mean())

    for treino, teste in gerador.split(np.zeros(len(y)), y, grupos):
        y_treino = y.iloc[treino]
        y_teste = y.iloc[teste]
        if y_treino.nunique() < 2 or y_teste.nunique() < 2:
            continue
        erro_tamanho = abs((len(teste) / len(y)) - test_size)
        erro_classe = abs(float(y_treino.mean()) - proporcao_global)
        erro_classe += abs(float(y_teste.mean()) - proporcao_global)
        nota = erro_tamanho + erro_classe
        if melhor is None or nota < melhor[0]:
            melhor = (nota, treino, teste)

    if melhor is None:
        raise ValueError(
            "Nao foi possivel criar treino e teste com as duas classes sem misturar "
            "o mesmo grupo. Verifique se existem partidas/grupos normais e hackers "
            "em quantidade suficiente."
        )
    return melhor[1], melhor[2]


def salvar_resultados(
    saida: Path,
    modelo: Pipeline,
    features: list[str],
    dados_teste: pd.DataFrame,
    y_teste: pd.Series,
    probabilidades: np.ndarray,
    previsoes: np.ndarray,
    metricas: dict[str, object],
    threshold: float,
) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    arvore: DecisionTreeClassifier = modelo.named_steps["arvore"]

    pacote = {
        "modelo": modelo,
        "features": features,
        "threshold": threshold,
        "classe_positiva": "hacker",
        "observacao": "Modelo de triagem/revisao; nao executar banimento automatico.",
    }
    joblib.dump(pacote, saida / "modelo_arvore_decisao.joblib")

    regras = export_text(arvore, feature_names=features, decimals=4)
    (saida / "regras_arvore.txt").write_text(regras, encoding="utf-8")

    importancias = pd.DataFrame(
        {"feature": features, "importancia": arvore.feature_importances_}
    ).sort_values("importancia", ascending=False)
    importancias.to_csv(saida / "importancia_features.csv", index=False)

    colunas_auditoria = [
        "arquivo_origem",
        "partida_id",
        "jogador_sessao",
        "janela_analise_10s",
        "inicio_janela_s",
        "fim_janela_s",
        "tipo_rotulo",
    ]
    previsoes_df = dados_teste[colunas_auditoria].copy()
    previsoes_df["rotulo_real"] = y_teste.to_numpy()
    previsoes_df["probabilidade_hacker"] = probabilidades
    previsoes_df["previsao_modelo"] = previsoes
    previsoes_df.to_csv(saida / "previsoes_teste.csv", index=False)

    (saida / "metricas_teste.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    altura = max(8.0, min(20.0, 2.2 * (arvore.get_depth() + 1)))
    plt.figure(figsize=(24, altura))
    plot_tree(
        arvore,
        feature_names=features,
        class_names=["normal", "hacker"],
        filled=True,
        rounded=True,
        proportion=True,
        precision=3,
        fontsize=8,
    )
    plt.title("Arvore de decisao - triagem comportamental anti-cheat")
    plt.tight_layout()
    plt.savefig(saida / "arvore_decisao.png", dpi=180, bbox_inches="tight")
    plt.close()


def treinar(args: argparse.Namespace) -> None:
    manifesto_path = Path(args.manifesto).resolve()
    manifesto = carregar_manifesto(manifesto_path)
    dados = carregar_intervalos_rotulados(manifesto, manifesto_path.parent)
    features = selecionar_features(dados)

    for coluna in features:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados[features] = dados[features].replace([np.inf, -np.inf], np.nan)
    totalmente_vazias = [coluna for coluna in features if dados[coluna].isna().all()]
    if totalmente_vazias:
        print(f"AVISO: features totalmente vazias e removidas: {totalmente_vazias}")
        features = [coluna for coluna in features if coluna not in totalmente_vazias]

    y = dados["rotulo_real"].astype(int)
    if y.nunique() < 2:
        raise ValueError("O manifesto precisa conter janelas normais e hackers.")

    if args.agrupar_por == "jogador_global":
        if (dados["jogador_global"].str.strip() == "").any():
            raise ValueError(
                "Para agrupar por jogador_global, preencha esse campo em todas as "
                "linhas do manifesto. Ele nunca sera usado como feature."
            )
        grupos = dados["jogador_global"].astype(str)
    else:
        grupos = dados["partida_id"].astype(str)

    idx_treino, idx_teste = escolher_split_por_grupo(
        y,
        grupos,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    treino = dados.iloc[idx_treino].copy()
    teste = dados.iloc[idx_teste].copy()
    X_treino = treino[features]
    X_teste = teste[features]
    y_treino = treino["rotulo_real"].astype(int)
    y_teste = teste["rotulo_real"].astype(int)

    min_folha = min(args.min_samples_leaf, max(2, len(treino) // 10))
    modelo = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            (
                "arvore",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=args.max_depth,
                    min_samples_leaf=min_folha,
                    min_samples_split=max(2 * min_folha, 4),
                    class_weight="balanced",
                    ccp_alpha=args.ccp_alpha,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    modelo.fit(X_treino, y_treino)

    probabilidades = modelo.predict_proba(X_teste)[:, 1]
    previsoes = (probabilidades >= args.threshold).astype(int)
    cm = confusion_matrix(y_teste, previsoes, labels=[0, 1])
    tn, fp, fn, tp = [int(valor) for valor in cm.ravel()]
    metricas: dict[str, object] = {
        "threshold": args.threshold,
        "grupo_validacao": args.agrupar_por,
        "grupos_treino": sorted(grupos.iloc[idx_treino].unique().tolist()),
        "grupos_teste": sorted(grupos.iloc[idx_teste].unique().tolist()),
        "amostras_total": int(len(dados)),
        "amostras_treino": int(len(treino)),
        "amostras_teste": int(len(teste)),
        "normais_teste": int((y_teste == 0).sum()),
        "hackers_teste": int((y_teste == 1).sum()),
        "accuracy": float(accuracy_score(y_teste, previsoes)),
        "balanced_accuracy": float(balanced_accuracy_score(y_teste, previsoes)),
        "precision_hacker": float(precision_score(y_teste, previsoes, zero_division=0)),
        "recall_hacker": float(recall_score(y_teste, previsoes, zero_division=0)),
        "f1_hacker": float(f1_score(y_teste, previsoes, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else math.nan,
        "matriz_confusao": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "profundidade_arvore": int(modelo.named_steps["arvore"].get_depth()),
        "folhas_arvore": int(modelo.named_steps["arvore"].get_n_leaves()),
        "features": features,
    }
    try:
        metricas["roc_auc"] = float(roc_auc_score(y_teste, probabilidades))
    except ValueError:
        metricas["roc_auc"] = None

    saida = Path(args.saida).resolve()
    salvar_resultados(
        saida,
        modelo,
        features,
        teste,
        y_teste,
        probabilidades,
        previsoes,
        metricas,
        args.threshold,
    )

    print("\n=== TREINAMENTO CONCLUIDO ===")
    print(f"Janelas rotuladas: {len(dados)}")
    print(f"Treino: {len(treino)} | Teste: {len(teste)}")
    print(f"Grupos de teste: {metricas['grupos_teste']}")
    print(f"Features usadas: {len(features)}")
    print(f"Profundidade efetiva: {metricas['profundidade_arvore']}")
    print(f"Precisao hacker: {metricas['precision_hacker']:.3f}")
    print(f"Recall hacker: {metricas['recall_hacker']:.3f}")
    print(f"F1 hacker: {metricas['f1_hacker']:.3f}")
    print(f"Taxa de falso positivo: {metricas['false_positive_rate']:.3f}")
    print("\nRelatorio detalhado:")
    print(
        classification_report(
            y_teste,
            previsoes,
            labels=[0, 1],
            target_names=["normal", "hacker"],
            zero_division=0,
        )
    )
    print(f"Arquivos salvos em: {saida}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Treina uma arvore explicavel com CSVs *_janelas.csv e rotulos "
            "independentes do motor de regras."
        )
    )
    parser.add_argument("--manifesto", help="CSV com intervalos e rotulos reais.")
    parser.add_argument(
        "--criar-template",
        metavar="ARQUIVO.csv",
        help="Cria um manifesto vazio e encerra.",
    )
    parser.add_argument(
        "--saida",
        default="models/arvore_decisao",
        help="Diretorio dos resultados (padrao: models/arvore_decisao).",
    )
    parser.add_argument(
        "--agrupar-por",
        choices=["partida_id", "jogador_global"],
        default="partida_id",
        help="Unidade mantida integralmente no treino ou teste.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--ccp-alpha", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()
    if args.criar_template:
        criar_template(Path(args.criar_template).resolve())
        return
    if not args.manifesto:
        parser.error("informe --manifesto ou --criar-template")
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size deve estar entre 0 e 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold deve estar entre 0 e 1")
    if args.max_depth < 1 or args.min_samples_leaf < 1 or args.ccp_alpha < 0:
        parser.error("hiperparametros da arvore precisam ser positivos")
    treinar(args)


if __name__ == "__main__":
    main()
