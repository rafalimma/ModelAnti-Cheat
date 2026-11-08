"""Treina e valida uma regressao logistica para triagem de ESP.

O script reutiliza o carregamento seguro, a lista branca de features, a
deteccao de datasets duplicados e a porta deterministica de 300 metros do
Random Forest. Use ``--logo`` para Leave-One-Group-Out; sem essa opcao, uma
divisao holdout por partida e executada e depois um modelo final e treinado
com todos os dados.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import trainer_arvore_decisao as arvore_base
import trainer_random_forest as rf_base


def criar_modelo(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("padronizador", StandardScaler()),
            (
                "regressao",
                LogisticRegression(
                    C=args.c,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                ),
            ),
        ]
    )


def tabela_coeficientes(modelo: Pipeline, features: list[str]) -> pd.DataFrame:
    regressao: LogisticRegression = modelo.named_steps["regressao"]
    coeficientes = regressao.coef_[0]
    tabela = pd.DataFrame(
        {
            "feature": features,
            "coeficiente_padronizado": coeficientes,
            "razao_chance": np.exp(np.clip(coeficientes, -50, 50)),
            "efeito": np.where(
                coeficientes > 0,
                "aumenta_suspeita",
                np.where(coeficientes < 0, "reduz_suspeita", "neutro"),
            ),
            "magnitude_absoluta": np.abs(coeficientes),
        }
    )
    return tabela.sort_values("magnitude_absoluta", ascending=False)


def preparar_dados(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], pd.Series, list[Path]]:
    manifestos = [Path(item).resolve() for item in args.manifesto]
    dados, _ = rf_base.carregar_dados_manifestos(
        manifestos,
        permitir_duplicatas=args.permitir_duplicatas,
    )
    features = rf_base.selecionar_features(dados, args.modo_features)

    for coluna in features:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados[features] = dados[features].replace([np.inf, -np.inf], np.nan)
    vazias = [coluna for coluna in features if dados[coluna].isna().all()]
    if vazias:
        print(f"AVISO: features totalmente vazias removidas: {vazias}")
        features = [coluna for coluna in features if coluna not in vazias]

    if dados["rotulo_real"].nunique() < 2:
        raise ValueError("Os manifestos precisam conter janelas normais e hackers.")

    if args.agrupar_por == "jogador_global":
        if (dados["jogador_global"].astype(str).str.strip() == "").any():
            raise ValueError("Preencha jogador_global em todos os manifestos.")
        grupos = dados["jogador_global"].astype(str)
    else:
        grupos = dados["partida_id"].astype(str)
    return dados, features, grupos, manifestos


def colunas_auditoria() -> list[str]:
    return [
        "arquivo_origem",
        "partida_id",
        "jogador_sessao",
        "janela_analise_10s",
        "inicio_janela_s",
        "fim_janela_s",
        "tipo_rotulo",
    ]


def pacote_modelo(
    modelo: Pipeline,
    features: list[str],
    args: argparse.Namespace,
    observacao: str,
) -> dict[str, object]:
    return {
        "modelo": modelo,
        "features": features,
        "threshold": args.threshold,
        "classe_positiva": "hacker",
        "modo_features": args.modo_features,
        "porta_esp": {
            "alvo_oculto_obrigatorio": True,
            "distancia_operador": ">",
            "distancia_minima_m": args.distancia_minima_esp,
        },
        "observacao": observacao,
    }


def executar_holdout(args: argparse.Namespace) -> None:
    dados, features, grupos, manifestos = preparar_dados(args)
    y = dados["rotulo_real"].astype(int)
    idx_treino, idx_teste = arvore_base.escolher_split_por_grupo(
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
    probabilidades, previsoes, elegiveis = rf_base.aplicar_porta_esp(
        modelo_avaliacao,
        teste,
        features,
        threshold=args.threshold,
        distancia_minima=args.distancia_minima_esp,
    )
    metricas = rf_base.calcular_metricas(y_teste, probabilidades, previsoes)
    metricas.update(
        {
            "metodo": "holdout_por_grupo",
            "grupo": args.agrupar_por,
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
            "hiperparametros": {
                "penalty": "l2",
                "C": args.c,
                "solver": "liblinear",
                "class_weight": "balanced",
                "threshold": args.threshold,
            },
        }
    )

    modelo_final = criar_modelo(args)
    modelo_final.fit(dados[features], y)
    saida = Path(args.saida).resolve()
    saida.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        pacote_modelo(
            modelo_final,
            features,
            args,
            "Modelo final treinado com todos os dados; usar como triagem.",
        ),
        saida / "modelo_regressao_logistica.joblib",
    )
    joblib.dump(
        pacote_modelo(
            modelo_avaliacao,
            features,
            args,
            "Modelo do holdout salvo para reproduzir as metricas.",
        ),
        saida / "modelo_regressao_logistica_holdout.joblib",
    )
    tabela_coeficientes(modelo_final, features).to_csv(
        saida / "coeficientes.csv", index=False
    )
    tabela_coeficientes(modelo_avaliacao, features).to_csv(
        saida / "coeficientes_holdout.csv", index=False
    )

    auditoria = teste[colunas_auditoria()].copy()
    auditoria["rotulo_real"] = y_teste.to_numpy()
    auditoria["elegivel_porta_esp"] = elegiveis.to_numpy()
    auditoria["probabilidade_hacker"] = probabilidades
    auditoria["previsao_modelo"] = previsoes
    auditoria.to_csv(saida / "previsoes_teste.csv", index=False)
    (saida / "metricas_teste.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cm = metricas["matriz_confusao"]
    print("\n=== REGRESSAO LOGISTICA CONCLUIDA ===")
    print(f"Partidas/grupos: {grupos.nunique()}")
    print(f"Janelas rotuladas: {len(dados)}")
    print(f"Treino: {len(treino)} | Teste: {len(teste)}")
    print(f"Grupos de teste: {metricas['grupos_teste']}")
    print(f"Features: {len(features)} ({args.modo_features})")
    print(f"Janelas enviadas ao juri: {int(elegiveis.sum())}/{len(teste)}")
    print(
        f"Matriz: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}"
    )
    print(f"Precisao hacker: {metricas['precision_hacker']:.3f}")
    print(f"Recall hacker: {metricas['recall_hacker']:.3f}")
    print(f"F1 hacker: {metricas['f1_hacker']:.3f}")
    print(f"Taxa de falso positivo: {metricas['false_positive_rate']:.4f}")
    print("Modelo final treinado com 100% das janelas rotuladas.")
    print(f"Resultados: {saida}")


def executar_logo(args: argparse.Namespace) -> None:
    dados, features, grupos_serie, manifestos = preparar_dados(args)
    grupos = sorted(grupos_serie.unique().tolist())
    if len(grupos) < 3:
        raise ValueError("LOGO exige pelo menos tres grupos independentes.")

    saida = Path(args.saida).resolve()
    saida.mkdir(parents=True, exist_ok=True)
    linhas_metricas: list[dict[str, object]] = []
    previsoes_folds: list[pd.DataFrame] = []
    coeficientes_folds: list[pd.DataFrame] = []

    print(f"\n=== LOGO REGRESSAO LOGISTICA: {len(grupos)} RODADAS ===")
    print(f"Grupos: {grupos}")
    print(f"Features: {len(features)} ({args.modo_features})")
    print(
        "Porta ESP: alvo oculto e distancia media "
        f"> {args.distancia_minima_esp:.1f} m"
    )

    for numero, grupo_teste in enumerate(grupos, start=1):
        mascara_teste = grupos_serie.eq(grupo_teste)
        treino = dados.loc[~mascara_teste].copy()
        teste = dados.loc[mascara_teste].copy()
        y_treino = treino["rotulo_real"].astype(int)
        y_teste = teste["rotulo_real"].astype(int)
        if y_treino.nunique() < 2:
            raise ValueError(
                f"Rodada {grupo_teste}: o treino ficou sem as duas classes."
            )

        modelo = criar_modelo(args)
        modelo.fit(treino[features], y_treino)
        probabilidades, previsoes, elegiveis = rf_base.aplicar_porta_esp(
            modelo,
            teste,
            features,
            threshold=args.threshold,
            distancia_minima=args.distancia_minima_esp,
        )
        metricas = rf_base.calcular_metricas(y_teste, probabilidades, previsoes)
        cm = metricas["matriz_confusao"]
        linhas_metricas.append(
            {
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
            }
        )

        auditoria = teste[colunas_auditoria()].copy()
        auditoria["grupo_teste"] = grupo_teste
        auditoria["rotulo_real"] = y_teste.to_numpy()
        auditoria["elegivel_porta_esp"] = elegiveis.to_numpy()
        auditoria["probabilidade_hacker"] = probabilidades
        auditoria["previsao_modelo"] = previsoes
        previsoes_folds.append(auditoria)

        tabela = tabela_coeficientes(modelo, features)
        tabela.insert(0, "grupo_teste", grupo_teste)
        tabela.insert(0, "rodada", numero)
        coeficientes_folds.append(tabela)

        print(
            f"[{numero}/{len(grupos)}] teste={grupo_teste}: "
            f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']} "
            f"| juri={int(elegiveis.sum())}/{len(teste)}"
        )

    previsoes_logo = pd.concat(previsoes_folds, ignore_index=True)
    y_logo = previsoes_logo["rotulo_real"].astype(int)
    proba_logo = previsoes_logo["probabilidade_hacker"].to_numpy()
    pred_logo = previsoes_logo["previsao_modelo"].astype(int).to_numpy()
    acumuladas = rf_base.calcular_metricas(y_logo, proba_logo, pred_logo)
    cm_total = acumuladas["matriz_confusao"]

    metricas_df = pd.DataFrame(linhas_metricas)
    metricas_df.to_csv(saida / "metricas_por_rodada.csv", index=False)
    previsoes_logo.to_csv(saida / "previsoes_logo.csv", index=False)

    coeficientes = pd.concat(coeficientes_folds, ignore_index=True)
    coeficientes.to_csv(saida / "coeficientes_por_rodada.csv", index=False)
    resumo_coef = (
        coeficientes.groupby("feature", as_index=False)
        .agg(
            coeficiente_medio=("coeficiente_padronizado", "mean"),
            coeficiente_std=("coeficiente_padronizado", "std"),
            magnitude_media=("magnitude_absoluta", "mean"),
        )
        .sort_values("magnitude_media", ascending=False)
    )
    resumo_coef["razao_chance_media_aproximada"] = np.exp(
        np.clip(resumo_coef["coeficiente_medio"], -50, 50)
    )
    resumo_coef.to_csv(saida / "coeficientes_media_logo.csv", index=False)

    partidas_com_fp = int(
        metricas_df.loc[metricas_df["fp"] > 0, "grupo_teste"].nunique()
    )
    partidas_hacker = int(
        metricas_df.loc[
            metricas_df["hackers_teste"] > 0, "grupo_teste"
        ].nunique()
    )
    partidas_hacker_detectadas = int(
        metricas_df.loc[metricas_df["tp"] > 0, "grupo_teste"].nunique()
    )
    resumo = {
        "metodo": "Leave-One-Group-Out",
        "algoritmo": "LogisticRegression",
        "grupo": args.agrupar_por,
        "modo_features": args.modo_features,
        "manifestos": [str(item) for item in manifestos],
        "rodadas": len(grupos),
        "janelas_total": int(len(previsoes_logo)),
        "janelas_enviadas_juri": int(
            previsoes_logo["elegivel_porta_esp"].sum()
        ),
        "partidas_com_falso_positivo": partidas_com_fp,
        "partidas_com_hacker": partidas_hacker,
        "partidas_hacker_detectadas": partidas_hacker_detectadas,
        "metricas_acumuladas": acumuladas,
        "hiperparametros": {
            "penalty": "l2",
            "C": args.c,
            "solver": "liblinear",
            "class_weight": "balanced",
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

    modelo_final = criar_modelo(args)
    modelo_final.fit(dados[features], dados["rotulo_real"].astype(int))
    joblib.dump(
        pacote_modelo(
            modelo_final,
            features,
            args,
            "Modelo final; metricas fora da amostra estao em resumo_logo.json.",
        ),
        saida / "modelo_regressao_logistica_logo_final.joblib",
    )
    tabela_coeficientes(modelo_final, features).to_csv(
        saida / "coeficientes_modelo_final.csv", index=False
    )

    print("\n=== LOGO REGRESSAO LOGISTICA CONCLUIDO ===")
    print(
        f"Matriz acumulada: TN={cm_total['tn']} FP={cm_total['fp']} "
        f"FN={cm_total['fn']} TP={cm_total['tp']}"
    )
    print(f"Precisao hacker: {acumuladas['precision_hacker']:.3f}")
    print(f"Recall hacker: {acumuladas['recall_hacker']:.3f}")
    print(f"F1 hacker: {acumuladas['f1_hacker']:.3f}")
    fpr = acumuladas["false_positive_rate"]
    print(f"Taxa de falso positivo: {fpr:.4f}" if not math.isnan(fpr) else "FPR: n/a")
    print(f"Partidas com falso positivo: {partidas_com_fp}/{len(grupos)}")
    print(
        "Partidas hacker detectadas: "
        f"{partidas_hacker_detectadas}/{partidas_hacker}"
    )
    print(f"Resultados: {saida}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina ou valida regressao logistica anti-cheat."
    )
    parser.add_argument(
        "--manifesto",
        action="append",
        required=True,
        help="Repita esta opcao para combinar varios manifestos.",
    )
    parser.add_argument("--saida", default="models/regressao_logistica")
    parser.add_argument("--logo", action="store_true", help="Executa LOGO.")
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
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--distancia-minima-esp", type=float, default=300.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--permitir-duplicatas", action="store_true")
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
    if args.c <= 0 or args.max_iter < 100:
        parser.error("--c deve ser positivo e --max-iter pelo menos 100")
    if args.logo:
        executar_logo(args)
    else:
        executar_holdout(args)


if __name__ == "__main__":
    main()
