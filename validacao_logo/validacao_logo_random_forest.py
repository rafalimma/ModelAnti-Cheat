"""Validacao Leave-One-Group-Out do Random Forest anti-cheat.

Cada rodada deixa uma partida inteira fora do treinamento. As previsoes de
todas as rodadas sao concatenadas para formar metricas genuinamente fora da
amostra. O script reutiliza o carregamento, a lista branca de features e a
porta deterministica do ``trainer_random_forest.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import trainer_random_forest as rf


def preparar_dados(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], pd.Series, list[Path]]:
    manifestos = [Path(item).resolve() for item in args.manifesto]
    dados, _ = rf.carregar_dados_manifestos(
        manifestos,
        permitir_duplicatas=args.permitir_duplicatas,
    )
    features = rf.selecionar_features(dados, args.modo_features)

    for coluna in features:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados[features] = dados[features].replace([np.inf, -np.inf], np.nan)
    vazias = [coluna for coluna in features if dados[coluna].isna().all()]
    if vazias:
        print(f"AVISO: features totalmente vazias removidas: {vazias}")
        features = [coluna for coluna in features if coluna not in vazias]

    if dados["rotulo_real"].nunique() < 2:
        raise ValueError("Os manifestos precisam conter as duas classes.")

    if args.agrupar_por == "jogador_global":
        if (dados["jogador_global"].astype(str).str.strip() == "").any():
            raise ValueError("Preencha jogador_global em todos os manifestos.")
        grupos = dados["jogador_global"].astype(str)
    else:
        grupos = dados["partida_id"].astype(str)

    if grupos.nunique() < 3:
        raise ValueError("LOGO exige pelo menos tres grupos independentes.")
    return dados, features, grupos, manifestos


def executar(args: argparse.Namespace) -> None:
    dados, features, grupos_serie, manifestos = preparar_dados(args)
    grupos = sorted(grupos_serie.unique().tolist())
    saida = Path(args.saida).resolve()
    saida.mkdir(parents=True, exist_ok=True)

    metricas_rodadas: list[dict[str, object]] = []
    previsoes_rodadas: list[pd.DataFrame] = []
    importancias_rodadas: list[pd.DataFrame] = []

    print(f"\n=== LOGO RANDOM FOREST: {len(grupos)} RODADAS ===")
    print(f"Grupos: {grupos}")
    print(f"Features: {len(features)} ({args.modo_features})")
    print("Porta ESP: alvo oculto e distancia media "
          f"> {args.distancia_minima_esp:.1f} m")

    for numero, grupo_teste in enumerate(grupos, start=1):
        mascara_teste = grupos_serie.eq(grupo_teste)
        treino = dados.loc[~mascara_teste].copy()
        teste = dados.loc[mascara_teste].copy()
        y_treino = treino["rotulo_real"].astype(int)
        y_teste = teste["rotulo_real"].astype(int)

        if y_treino.nunique() < 2:
            raise ValueError(
                f"Rodada {grupo_teste}: o treino ficou sem as duas classes. "
                "Sao necessarias acoes hacker em mais de uma partida."
            )

        modelo = rf.criar_modelo(args)
        modelo.fit(treino[features], y_treino)
        probabilidades, previsoes, elegiveis = rf.aplicar_porta_esp(
            modelo,
            teste,
            features,
            threshold=args.threshold,
            distancia_minima=args.distancia_minima_esp,
        )

        metricas = rf.calcular_metricas(y_teste, probabilidades, previsoes)
        cm = metricas["matriz_confusao"]
        linha = {
            "rodada": numero,
            "grupo_teste": grupo_teste,
            "janelas_teste": int(len(teste)),
            "normais_teste": int((y_teste == 0).sum()),
            "hackers_teste": int((y_teste == 1).sum()),
            "janelas_enviadas_juri": int(elegiveis.sum()),
            "tn": int(cm["tn"]),
            "fp": int(cm["fp"]),
            "fn": int(cm["fn"]),
            "tp": int(cm["tp"]),
            "precision_hacker": metricas["precision_hacker"],
            "recall_hacker": metricas["recall_hacker"],
            "f1_hacker": metricas["f1_hacker"],
            "false_positive_rate": metricas["false_positive_rate"],
            "roc_auc": metricas["roc_auc"],
            "average_precision": metricas["average_precision"],
            "oob_score_treino": float(modelo.named_steps["floresta"].oob_score_),
        }
        metricas_rodadas.append(linha)

        auditoria = teste[
            [
                "arquivo_origem",
                "partida_id",
                "jogador_sessao",
                "janela_analise_10s",
                "inicio_janela_s",
                "fim_janela_s",
                "tipo_rotulo",
            ]
        ].copy()
        auditoria["grupo_teste"] = grupo_teste
        auditoria["rotulo_real"] = y_teste.to_numpy()
        auditoria["elegivel_porta_esp"] = elegiveis.to_numpy()
        auditoria["probabilidade_hacker"] = probabilidades
        auditoria["previsao_modelo"] = previsoes
        previsoes_rodadas.append(auditoria)

        importancias_rodadas.append(
            pd.DataFrame(
                {
                    "rodada": numero,
                    "grupo_teste": grupo_teste,
                    "feature": features,
                    "importancia": modelo.named_steps[
                        "floresta"
                    ].feature_importances_,
                }
            )
        )

        print(
            f"[{numero}/{len(grupos)}] teste={grupo_teste}: "
            f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']} "
            f"| juri={int(elegiveis.sum())}/{len(teste)}"
        )

    previsoes_logo = pd.concat(previsoes_rodadas, ignore_index=True)
    y_logo = previsoes_logo["rotulo_real"].astype(int)
    proba_logo = previsoes_logo["probabilidade_hacker"].to_numpy()
    pred_logo = previsoes_logo["previsao_modelo"].astype(int).to_numpy()
    metricas_acumuladas = rf.calcular_metricas(y_logo, proba_logo, pred_logo)
    cm_total = metricas_acumuladas["matriz_confusao"]

    tabela_metricas = pd.DataFrame(metricas_rodadas)
    tabela_metricas.to_csv(saida / "metricas_por_rodada.csv", index=False)
    previsoes_logo.to_csv(saida / "previsoes_logo.csv", index=False)

    importancias = pd.concat(importancias_rodadas, ignore_index=True)
    importancias.to_csv(saida / "importancia_features_por_rodada.csv", index=False)
    resumo_importancias = (
        importancias.groupby("feature", as_index=False)["importancia"]
        .agg(importancia_media="mean", importancia_std="std")
        .sort_values("importancia_media", ascending=False)
    )
    resumo_importancias.to_csv(
        saida / "importancia_features_media_logo.csv", index=False
    )

    partidas_com_fp = int(
        tabela_metricas.loc[tabela_metricas["fp"] > 0, "grupo_teste"].nunique()
    )
    partidas_com_hacker = int(
        tabela_metricas.loc[
            tabela_metricas["hackers_teste"] > 0, "grupo_teste"
        ].nunique()
    )
    partidas_hacker_detectadas = int(
        tabela_metricas.loc[tabela_metricas["tp"] > 0, "grupo_teste"].nunique()
    )

    resumo = {
        "metodo": "Leave-One-Group-Out",
        "algoritmo": "RandomForestClassifier",
        "grupo": args.agrupar_por,
        "modo_features": args.modo_features,
        "manifestos": [str(item) for item in manifestos],
        "rodadas": len(grupos),
        "janelas_total": int(len(previsoes_logo)),
        "janelas_enviadas_juri": int(
            previsoes_logo["elegivel_porta_esp"].sum()
        ),
        "partidas_com_falso_positivo": partidas_com_fp,
        "partidas_com_hacker": partidas_com_hacker,
        "partidas_hacker_detectadas": partidas_hacker_detectadas,
        "metricas_acumuladas": metricas_acumuladas,
        "hiperparametros": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "class_weight": "balanced_subsample",
            "threshold": args.threshold,
        },
        "porta_esp": {
            "alvo_oculto_obrigatorio": True,
            "distancia_operador": ">",
            "distancia_minima_m": args.distancia_minima_esp,
        },
    }
    (saida / "resumo_logo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # O modelo operacional final usa todos os dados depois que a avaliacao
    # fora da amostra foi concluida.
    modelo_final = rf.criar_modelo(args)
    modelo_final.fit(dados[features], dados["rotulo_real"].astype(int))
    pacote = {
        "modelo": modelo_final,
        "features": features,
        "threshold": args.threshold,
        "classe_positiva": "hacker",
        "modo_features": args.modo_features,
        "porta_esp": resumo["porta_esp"],
        "observacao": "Modelo final; metricas validas estao em resumo_logo.json.",
    }
    joblib.dump(pacote, saida / "modelo_random_forest_logo_final.joblib")

    print("\n=== LOGO RANDOM FOREST CONCLUIDO ===")
    print(
        "Matriz acumulada: "
        f"TN={cm_total['tn']} FP={cm_total['fp']} "
        f"FN={cm_total['fn']} TP={cm_total['tp']}"
    )
    print(f"Precisao hacker: {metricas_acumuladas['precision_hacker']:.3f}")
    print(f"Recall hacker: {metricas_acumuladas['recall_hacker']:.3f}")
    print(f"F1 hacker: {metricas_acumuladas['f1_hacker']:.3f}")
    fpr = metricas_acumuladas["false_positive_rate"]
    print(f"Taxa de falso positivo: {fpr:.4f}" if not math.isnan(fpr) else "FPR: n/a")
    print(
        "Partidas com falso positivo: "
        f"{partidas_com_fp}/{len(grupos)}"
    )
    print(
        "Partidas hacker detectadas: "
        f"{partidas_hacker_detectadas}/{partidas_com_hacker}"
    )
    print(f"Resultados: {saida}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa LOGO do Random Forest anti-cheat."
    )
    parser.add_argument(
        "--manifesto",
        action="append",
        required=True,
        help="Repita a opcao para combinar varios manifestos.",
    )
    parser.add_argument("--saida", default="models/random_forest_logo")
    parser.add_argument(
        "--agrupar-por",
        choices=["partida_id", "jogador_global"],
        default="partida_id",
    )
    parser.add_argument(
        "--modo-features",
        choices=["auto", "comum", "contexto"],
        default="auto",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--distancia-minima-esp", type=float, default=300.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--permitir-duplicatas", action="store_true")
    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold deve estar entre 0 e 1")
    if args.distancia_minima_esp < 0:
        parser.error("--distancia-minima-esp nao pode ser negativa")
    if args.n_estimators < 10:
        parser.error("--n-estimators deve ser pelo menos 10")
    if args.max_depth < 1 or args.min_samples_leaf < 1:
        parser.error("hiperparametros devem ser positivos")
    executar(args)


if __name__ == "__main__":
    main()
