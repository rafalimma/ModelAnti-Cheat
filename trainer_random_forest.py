"""Treina um Random Forest para triagem comportamental anti-cheat.

O carregamento dos rotulos e a lista branca de features sao reutilizados do
treinador da arvore de decisao. Isso mantem a comparacao entre os modelos
justa e evita usar IDs, armas, pontuacoes ou decisoes do pre-processador como
features.

O script aceita mais de um manifesto. Exemplo combinado:

    python trainer_random_forest.py \
        --manifesto partidas_rotuladas/rotulos_treino_todas.csv \
        --manifesto partidas_rotuladas/rotulos_treino_par1plyr2_normal_contexto.csv \
        --manifesto partidas_rotuladas/rotulos_treino_par2plyr3_esphack_contexto.csv \
        --modo-features comum \
        --saida models/random_forest_combinado

Para um experimento somente com logs de contexto rico, use
``--modo-features contexto`` e informe apenas os manifestos novos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "model_anticheat_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

import trainer_arvore_decisao as base


DISTANCIA_MINIMA_ESP_PADRAO = 300.0


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def carregar_dados_manifestos(
    caminhos: list[Path],
    permitir_duplicatas: bool,
) -> tuple[pd.DataFrame, list[str]]:
    partes: list[pd.DataFrame] = []
    fontes: set[Path] = set()

    for caminho in caminhos:
        manifesto = base.carregar_manifesto(caminho)
        dados = base.carregar_intervalos_rotulados(manifesto, caminho.parent)
        dados["manifesto_origem"] = str(caminho)
        partes.append(dados)
        fontes.update(Path(item).resolve() for item in dados["arquivo_origem"].unique())

    dados = pd.concat(partes, ignore_index=True)
    chave = ["arquivo_origem", "jogador_sessao", "janela_analise_10s"]

    conflitos = dados.groupby(chave, dropna=False)["rotulo_real"].nunique()
    conflitos = conflitos[conflitos > 1]
    if not conflitos.empty:
        raise ValueError(
            "Manifestos diferentes atribuem rotulos conflitantes a uma janela. "
            f"Primeiro conflito: {conflitos.index[0]}"
        )

    antes = len(dados)
    dados = dados.drop_duplicates(chave, keep="first").reset_index(drop=True)
    if len(dados) < antes:
        print(
            "AVISO: janelas repetidas entre manifestos foram consolidadas: "
            f"{antes - len(dados)}"
        )

    hashes: dict[str, list[Path]] = {}
    for fonte in sorted(fontes):
        hashes.setdefault(sha256_arquivo(fonte), []).append(fonte)

    grupos_duplicados = [arquivos for arquivos in hashes.values() if len(arquivos) > 1]
    avisos: list[str] = []
    for arquivos in grupos_duplicados:
        descricao = " = ".join(str(item) for item in arquivos)
        avisos.append(descricao)

    if avisos and not permitir_duplicatas:
        detalhes = "\n  ".join(avisos)
        raise ValueError(
            "Foram encontrados CSVs de janelas com conteudo identico. Isso causaria "
            "vazamento entre treino e teste. Remova uma copia ou use "
            "--permitir-duplicatas apenas para diagnostico:\n  " + detalhes
        )
    if avisos:
        print("AVISO: datasets duplicados permitidos explicitamente:")
        for aviso in avisos:
            print(f"  {aviso}")

    return dados, avisos


def contexto_confiavel_em_todas_as_linhas(dados: pd.DataFrame) -> bool:
    return (
        "contexto_alvo_confiavel" in dados.columns
        and pd.to_numeric(
            dados["contexto_alvo_confiavel"], errors="coerce"
        ).fillna(0).eq(1).all()
    )


def selecionar_features(dados: pd.DataFrame, modo: str) -> list[str]:
    contexto_confiavel = contexto_confiavel_em_todas_as_linhas(dados)

    if modo == "contexto" and not contexto_confiavel:
        raise ValueError(
            "--modo-features contexto exige somente partidas geradas com o "
            "contexto confiavel do init.c novo."
        )

    usar_contexto = contexto_confiavel if modo == "auto" else modo == "contexto"
    candidatas = [
        coluna
        for coluna in base.FEATURES_COMPORTAMENTAIS
        if usar_contexto or coluna not in base.FEATURES_CONTEXTO_ALVO
    ]
    features = [coluna for coluna in candidatas if coluna in dados.columns]
    faltantes = [coluna for coluna in candidatas if coluna not in dados.columns]

    if faltantes:
        print(f"AVISO: features ausentes e ignoradas: {faltantes}")
    if not usar_contexto:
        print(
            "Features de contexto do mesmo alvo desativadas. "
            "O modelo usara somente o conjunto comum entre as geracoes."
        )

    vazamento = sorted(set(features) & base.COLUNAS_PROIBIDAS)
    if vazamento:
        raise AssertionError(f"Features proibidas entraram no modelo: {vazamento}")
    if len(features) < 10:
        raise ValueError(f"Somente {len(features)} features validas foram encontradas.")
    return features


def criar_modelo(args: argparse.Namespace) -> Pipeline:
    floresta = RandomForestClassifier(
        n_estimators=args.n_estimators,
        criterion="gini",
        max_depth=args.max_depth,
        min_samples_split=max(2 * args.min_samples_leaf, 4),
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=True,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    return Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("floresta", floresta),
        ]
    )


def aplicar_porta_esp(
    modelo: Pipeline,
    teste: pd.DataFrame,
    features: list[str],
    threshold: float,
    distancia_minima: float,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    requeridas = {"distancia_alvo_media", "taxa_alvo_oculto"}
    faltantes = sorted(requeridas - set(teste.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes para a porta de ESP: {faltantes}")

    distancia = pd.to_numeric(teste["distancia_alvo_media"], errors="coerce")
    taxa_oculta = pd.to_numeric(teste["taxa_alvo_oculto"], errors="coerce")
    elegiveis = distancia.gt(distancia_minima) & taxa_oculta.gt(0.0)

    probabilidades = np.zeros(len(teste), dtype=float)
    previsoes = np.zeros(len(teste), dtype=int)
    if elegiveis.any():
        proba_elegivel = modelo.predict_proba(teste.loc[elegiveis, features])[:, 1]
        probabilidades[np.flatnonzero(elegiveis.to_numpy())] = proba_elegivel
        previsoes[np.flatnonzero(elegiveis.to_numpy())] = (
            proba_elegivel >= threshold
        ).astype(int)
    return probabilidades, previsoes, elegiveis


def calcular_metricas(
    y_real: pd.Series,
    probabilidades: np.ndarray,
    previsoes: np.ndarray,
) -> dict[str, object]:
    cm = confusion_matrix(y_real, previsoes, labels=[0, 1])
    tn, fp, fn, tp = [int(valor) for valor in cm.ravel()]
    duas_classes = pd.Series(y_real).nunique() == 2
    metricas: dict[str, object] = {
        "accuracy": float(accuracy_score(y_real, previsoes)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_real, previsoes))
            if duas_classes
            else None
        ),
        "precision_hacker": float(
            precision_score(y_real, previsoes, zero_division=0)
        ),
        "recall_hacker": float(recall_score(y_real, previsoes, zero_division=0)),
        "f1_hacker": float(f1_score(y_real, previsoes, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else math.nan,
        "matriz_confusao": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    try:
        if not duas_classes:
            raise ValueError("A metrica exige as duas classes.")
        metricas["roc_auc"] = float(roc_auc_score(y_real, probabilidades))
        metricas["average_precision"] = float(
            average_precision_score(y_real, probabilidades)
        )
    except ValueError:
        metricas["roc_auc"] = None
        metricas["average_precision"] = None
    return metricas


def salvar_resultados(
    saida: Path,
    modelo_avaliacao: Pipeline,
    modelo_final: Pipeline,
    features: list[str],
    teste: pd.DataFrame,
    y_teste: pd.Series,
    probabilidades: np.ndarray,
    previsoes: np.ndarray,
    elegiveis: pd.Series,
    metricas: dict[str, object],
    args: argparse.Namespace,
) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    floresta_avaliacao: RandomForestClassifier = (
        modelo_avaliacao.named_steps["floresta"]
    )
    floresta_final: RandomForestClassifier = modelo_final.named_steps["floresta"]

    pacote_final = {
        "modelo": modelo_final,
        "features": features,
        "threshold": args.threshold,
        "classe_positiva": "hacker",
        "porta_esp": {
            "alvo_oculto_obrigatorio": True,
            "distancia_operador": ">",
            "distancia_minima_m": args.distancia_minima_esp,
        },
        "modo_features": args.modo_features,
        "observacao": "Modelo de triagem; nao executar banimento automatico.",
    }
    joblib.dump(pacote_final, saida / "modelo_random_forest.joblib")

    pacote_avaliacao = dict(pacote_final)
    pacote_avaliacao["modelo"] = modelo_avaliacao
    pacote_avaliacao["observacao"] = (
        "Modelo do holdout, salvo para reproduzir as metricas de teste."
    )
    joblib.dump(
        pacote_avaliacao,
        saida / "modelo_random_forest_holdout.joblib",
    )

    importancias = pd.DataFrame(
        {"feature": features, "importancia": floresta_final.feature_importances_}
    ).sort_values("importancia", ascending=False)
    importancias.to_csv(saida / "importancia_features.csv", index=False)

    pd.DataFrame(
        {
            "feature": features,
            "importancia": floresta_avaliacao.feature_importances_,
        }
    ).sort_values("importancia", ascending=False).to_csv(
        saida / "importancia_features_holdout.csv",
        index=False,
    )

    colunas_auditoria = [
        "arquivo_origem",
        "partida_id",
        "jogador_sessao",
        "janela_analise_10s",
        "inicio_janela_s",
        "fim_janela_s",
        "tipo_rotulo",
    ]
    auditoria = teste[colunas_auditoria].copy()
    auditoria["rotulo_real"] = y_teste.to_numpy()
    auditoria["elegivel_porta_esp"] = elegiveis.to_numpy()
    auditoria["probabilidade_hacker"] = probabilidades
    auditoria["previsao_modelo"] = previsoes
    auditoria.to_csv(saida / "previsoes_teste.csv", index=False)

    elegiveis_array = elegiveis.to_numpy()
    if elegiveis_array.sum() >= 10 and y_teste.iloc[elegiveis_array].nunique() > 1:
        resultado = permutation_importance(
            modelo_avaliacao,
            teste.loc[elegiveis, features],
            y_teste.loc[elegiveis],
            scoring="average_precision",
            n_repeats=10,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        pd.DataFrame(
            {
                "feature": features,
                "importancia_media": resultado.importances_mean,
                "importancia_std": resultado.importances_std,
            }
        ).sort_values("importancia_media", ascending=False).to_csv(
            saida / "importancia_permutacao_teste.csv", index=False
        )

    metricas["oob_score_holdout"] = float(floresta_avaliacao.oob_score_)
    metricas["oob_score_modelo_final"] = float(floresta_final.oob_score_)
    (saida / "metricas_teste.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top = importancias.head(20).sort_values("importancia")
    plt.figure(figsize=(10, max(6, len(top) * 0.35)))
    plt.barh(top["feature"], top["importancia"], color="#2F6B4F")
    plt.xlabel("Importancia media por reducao de impureza")
    plt.title("Random Forest - importancia das features")
    plt.tight_layout()
    plt.savefig(saida / "importancia_features.png", dpi=180, bbox_inches="tight")
    plt.close()


def treinar(args: argparse.Namespace) -> None:
    manifestos = [Path(item).resolve() for item in args.manifesto]
    dados, avisos_duplicatas = carregar_dados_manifestos(
        manifestos,
        permitir_duplicatas=args.permitir_duplicatas,
    )
    features = selecionar_features(dados, args.modo_features)

    for coluna in features:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados[features] = dados[features].replace([np.inf, -np.inf], np.nan)
    totalmente_vazias = [coluna for coluna in features if dados[coluna].isna().all()]
    if totalmente_vazias:
        print(f"AVISO: features totalmente vazias removidas: {totalmente_vazias}")
        features = [coluna for coluna in features if coluna not in totalmente_vazias]

    y = dados["rotulo_real"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Os manifestos precisam conter janelas normais e hackers.")

    if args.agrupar_por == "jogador_global":
        if (dados["jogador_global"].astype(str).str.strip() == "").any():
            raise ValueError("Preencha jogador_global em todos os manifestos.")
        grupos = dados["jogador_global"].astype(str)
    else:
        grupos = dados["partida_id"].astype(str)

    idx_treino, idx_teste = base.escolher_split_por_grupo(
        y,
        grupos,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    treino = dados.iloc[idx_treino].copy()
    teste = dados.iloc[idx_teste].copy()
    y_treino = treino["rotulo_real"].astype(int)
    y_teste = teste["rotulo_real"].astype(int)

    modelo_avaliacao = criar_modelo(args)
    modelo_avaliacao.fit(treino[features], y_treino)
    probabilidades, previsoes, elegiveis = aplicar_porta_esp(
        modelo_avaliacao,
        teste,
        features,
        threshold=args.threshold,
        distancia_minima=args.distancia_minima_esp,
    )

    metricas = calcular_metricas(y_teste, probabilidades, previsoes)
    metricas.update(
        {
            "threshold": args.threshold,
            "grupo_validacao": args.agrupar_por,
            "grupos_treino": sorted(grupos.iloc[idx_treino].unique().tolist()),
            "grupos_teste": sorted(grupos.iloc[idx_teste].unique().tolist()),
            "amostras_total": int(len(dados)),
            "amostras_treino": int(len(treino)),
            "amostras_teste": int(len(teste)),
            "normais_teste": int((y_teste == 0).sum()),
            "hackers_teste": int((y_teste == 1).sum()),
            "janelas_enviadas_juri": int(elegiveis.sum()),
            "features": features,
            "modo_features": args.modo_features,
            "manifestos": [str(item) for item in manifestos],
            "duplicatas_permitidas": avisos_duplicatas,
            "hiperparametros": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "max_features": args.max_features,
                "class_weight": "balanced_subsample",
            },
            "porta_esp": {
                "alvo_oculto_obrigatorio": True,
                "distancia_operador": ">",
                "distancia_minima_m": args.distancia_minima_esp,
            },
        }
    )

    # Depois de medir o desempenho fora da amostra, este segundo modelo usa
    # todas as partidas rotuladas. Ele e o artefato indicado para inferencia;
    # o modelo do holdout permanece salvo apenas para reproduzir as metricas.
    modelo_final = criar_modelo(args)
    modelo_final.fit(dados[features], y)

    saida = Path(args.saida).resolve()
    salvar_resultados(
        saida,
        modelo_avaliacao,
        modelo_final,
        features,
        teste,
        y_teste,
        probabilidades,
        previsoes,
        elegiveis,
        metricas,
        args,
    )

    print("\n=== RANDOM FOREST CONCLUIDO ===")
    print(f"Manifestos: {len(manifestos)}")
    print(f"Partidas/grupos: {grupos.nunique()}")
    print(f"Janelas rotuladas: {len(dados)}")
    print(f"Treino: {len(treino)} | Teste: {len(teste)}")
    print(f"Grupos de teste: {metricas['grupos_teste']}")
    print(f"Features usadas: {len(features)} ({args.modo_features})")
    print(f"Janelas enviadas ao juri: {int(elegiveis.sum())}/{len(teste)}")
    print(f"Precisao hacker: {metricas['precision_hacker']:.3f}")
    print(f"Recall hacker: {metricas['recall_hacker']:.3f}")
    print(f"F1 hacker: {metricas['f1_hacker']:.3f}")
    print(f"Taxa de falso positivo: {metricas['false_positive_rate']:.4f}")
    print(
        "OOB score do holdout: "
        f"{modelo_avaliacao.named_steps['floresta'].oob_score_:.3f}"
    )
    print(
        "OOB score do modelo final: "
        f"{modelo_final.named_steps['floresta'].oob_score_:.3f}"
    )
    print("Modelo final treinado com 100% das janelas rotuladas.")
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
    print(f"Resultados: {saida}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina Random Forest com grupos independentes e porta de ESP."
    )
    parser.add_argument(
        "--manifesto",
        action="append",
        required=True,
        help="Manifesto de rotulos. Repita a opcao para combinar varios arquivos.",
    )
    parser.add_argument(
        "--saida",
        default="models/random_forest",
        help="Diretorio dos resultados.",
    )
    parser.add_argument(
        "--agrupar-por",
        choices=["partida_id", "jogador_global"],
        default="partida_id",
    )
    parser.add_argument(
        "--modo-features",
        choices=["auto", "comum", "contexto"],
        default="auto",
        help=(
            "comum combina logs antigos e novos; contexto usa as features novas; "
            "auto habilita contexto somente se todas as linhas forem confiaveis."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--distancia-minima-esp", type=float, default=300.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--permitir-duplicatas",
        action="store_true",
        help="Permite CSVs identicos; use somente para diagnostico.",
    )
    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size deve estar entre 0 e 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold deve estar entre 0 e 1")
    if args.distancia_minima_esp < 0:
        parser.error("--distancia-minima-esp nao pode ser negativa")
    if args.n_estimators < 10:
        parser.error("--n-estimators deve ser pelo menos 10")
    if args.max_depth < 1 or args.min_samples_leaf < 1:
        parser.error("--max-depth e --min-samples-leaf devem ser positivos")
    treinar(args)


if __name__ == "__main__":
    main()
