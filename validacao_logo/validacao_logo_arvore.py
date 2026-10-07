"""Valida a arvore anti-cheat deixando uma partida inteira fora por rodada.

Este experimento reutiliza a leitura do manifesto e a lista branca de features
do trainer principal, mas nao altera o seu comportamento. Em cada rodada, um
modelo novo e treinado nas demais partidas e avaliado apenas na partida que
ficou de fora (Leave-One-Group-Out).

As colunas de arma podem aparecer nos arquivos de auditoria, mas nunca entram
no treinamento: somente FEATURES_COMPORTAMENTAIS do trainer principal sao
usadas como entrada do modelo.
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

PASTA_SCRIPT = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_SCRIPT.parent
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

import trainer_arvore_decisao as base

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def preparar_dados(manifesto_path: Path) -> tuple[pd.DataFrame, list[str]]:
    manifesto = base.carregar_manifesto(manifesto_path)
    dados = base.carregar_intervalos_rotulados(manifesto, manifesto_path.parent)
    features = base.selecionar_features(dados)

    for coluna in features:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados[features] = dados[features].replace([np.inf, -np.inf], np.nan)

    totalmente_vazias = [coluna for coluna in features if dados[coluna].isna().all()]
    if totalmente_vazias:
        print(f"AVISO: features totalmente vazias e removidas: {totalmente_vazias}")
        features = [coluna for coluna in features if coluna not in totalmente_vazias]

    if dados["rotulo_real"].nunique() < 2:
        raise ValueError("O conjunto completo precisa conter janelas normais e hackers.")
    return dados, features


def criar_modelo(args: argparse.Namespace, tamanho_treino: int) -> Pipeline:
    min_folha = min(args.min_samples_leaf, max(2, tamanho_treino // 10))
    return Pipeline(
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


def dividir_metricas(
    y_real: pd.Series,
    previsoes: np.ndarray,
    probabilidades: np.ndarray,
) -> dict[str, object]:
    cm = confusion_matrix(y_real, previsoes, labels=[0, 1])
    tn, fp, fn, tp = [int(valor) for valor in cm.ravel()]
    tem_normal = (tn + fp) > 0
    tem_hacker = (tp + fn) > 0
    previu_hacker = (tp + fp) > 0
    duas_classes = y_real.nunique() == 2

    metricas: dict[str, object] = {
        "amostras_teste": int(len(y_real)),
        "normais_teste": int((y_real == 0).sum()),
        "hackers_teste": int((y_real == 1).sum()),
        "accuracy": float(accuracy_score(y_real, previsoes)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_real, previsoes)) if duas_classes else None
        ),
        "precision_hacker": (
            float(precision_score(y_real, previsoes, zero_division=0))
            if previu_hacker
            else None
        ),
        "recall_hacker": (
            float(recall_score(y_real, previsoes, zero_division=0))
            if tem_hacker
            else None
        ),
        "f1_hacker": (
            float(f1_score(y_real, previsoes, zero_division=0))
            if tem_hacker and previu_hacker
            else None
        ),
        "false_positive_rate": float(fp / (fp + tn)) if tem_normal else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "matriz_confusao": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    metricas["roc_auc"] = (
        float(roc_auc_score(y_real, probabilidades)) if duas_classes else None
    )
    return metricas


def montar_previsoes_auditoria(
    teste: pd.DataFrame,
    y_teste: pd.Series,
    probabilidades: np.ndarray,
    previsoes: np.ndarray,
    grupo_teste: str,
) -> pd.DataFrame:
    colunas = [
        "arquivo_origem",
        "partida_id",
        "jogador_sessao",
        "janela_analise_10s",
        "inicio_janela_s",
        "fim_janela_s",
        "tipo_rotulo",
    ]
    # Arma e apenas contexto para conferir rotulos; nao e feature do modelo.
    colunas += [
        coluna
        for coluna in ("arma_predominante", "armas_observadas")
        if coluna in teste.columns
    ]
    saida = teste[colunas].copy()
    saida.insert(0, "grupo_deixado_de_fora", grupo_teste)
    saida["rotulo_real"] = y_teste.to_numpy()
    saida["probabilidade_hacker"] = probabilidades
    saida["previsao_modelo"] = previsoes
    saida["acerto"] = saida["rotulo_real"] == saida["previsao_modelo"]
    return saida


def salvar_modelo_final(
    saida: Path,
    modelo: Pipeline,
    features: list[str],
    args: argparse.Namespace,
    grupos: list[str],
    amostras: int,
) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    arvore: DecisionTreeClassifier = modelo.named_steps["arvore"]
    pacote = {
        "modelo": modelo,
        "features": features,
        "threshold": args.threshold,
        "classe_positiva": "hacker",
        "grupos_treinamento": grupos,
        "observacao": "Modelo final de triagem/revisao; nao banir automaticamente.",
    }
    joblib.dump(pacote, saida / "modelo_arvore_decisao.joblib")
    (saida / "regras_arvore.txt").write_text(
        export_text(arvore, feature_names=features, decimals=4), encoding="utf-8"
    )
    pd.DataFrame(
        {"feature": features, "importancia": arvore.feature_importances_}
    ).sort_values("importancia", ascending=False).to_csv(
        saida / "importancia_features.csv", index=False
    )
    metadados = {
        "tipo": "modelo_final_treinado_com_todas_as_partidas",
        "nao_e_avaliacao_fora_da_amostra": True,
        "partidas_treinamento": grupos,
        "amostras_treinamento": amostras,
        "features": features,
        "arma_usada_como_feature": False,
        "profundidade_arvore": int(arvore.get_depth()),
        "folhas_arvore": int(arvore.get_n_leaves()),
    }
    (saida / "metadados_modelo_final.json").write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    altura = max(8.0, min(20.0, 2.2 * (arvore.get_depth() + 1)))
    base.plt.figure(figsize=(24, altura))
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
    base.plt.title("Arvore final - treinada com todas as partidas")
    base.plt.tight_layout()
    base.plt.savefig(saida / "arvore_decisao.png", dpi=180, bbox_inches="tight")
    base.plt.close()


def media_desvio(metricas_df: pd.DataFrame, coluna: str) -> dict[str, object]:
    valores = pd.to_numeric(metricas_df[coluna], errors="coerce").dropna()
    if valores.empty:
        return {"media": None, "desvio_padrao": None, "rodadas_validas": 0}
    return {
        "media": float(valores.mean()),
        "desvio_padrao": float(valores.std(ddof=0)),
        "rodadas_validas": int(len(valores)),
    }


def executar(args: argparse.Namespace) -> None:
    manifesto_path = Path(args.manifesto).resolve()
    saida = Path(args.saida).resolve()
    saida.mkdir(parents=True, exist_ok=True)
    dados, features = preparar_dados(manifesto_path)

    if args.agrupar_por == "jogador_global":
        if (dados["jogador_global"].astype(str).str.strip() == "").any():
            raise ValueError("Preencha jogador_global para agrupar por jogador.")
        grupos_serie = dados["jogador_global"].astype(str)
    else:
        grupos_serie = dados["partida_id"].astype(str)

    grupos = sorted(grupos_serie.unique().tolist())
    if len(grupos) < 3:
        raise ValueError("A validacao LOGO exige pelo menos 3 grupos independentes.")

    linhas_metricas: list[dict[str, object]] = []
    previsoes_fora_amostra: list[pd.DataFrame] = []
    importancias: list[pd.DataFrame] = []

    print(f"\n=== VALIDACAO LEAVE-ONE-GROUP-OUT: {len(grupos)} RODADAS ===")
    print(f"Grupos: {grupos}")
    print(f"Features comportamentais: {len(features)} (arma excluida)")

    for numero, grupo_teste in enumerate(grupos, start=1):
        mascara_teste = grupos_serie == grupo_teste
        treino = dados.loc[~mascara_teste].copy()
        teste = dados.loc[mascara_teste].copy()
        y_treino = treino["rotulo_real"].astype(int)
        y_teste = teste["rotulo_real"].astype(int)

        if y_treino.nunique() < 2:
            raise ValueError(
                f"Rodada {grupo_teste}: o treino ficou sem as duas classes. "
                "Sao necessarias acoes hacker em mais de uma partida."
            )

        modelo = criar_modelo(args, len(treino))
        modelo.fit(treino[features], y_treino)
        probabilidades = modelo.predict_proba(teste[features])[:, 1]
        previsoes = (probabilidades >= args.threshold).astype(int)
        metricas = dividir_metricas(y_teste, previsoes, probabilidades)
        arvore: DecisionTreeClassifier = modelo.named_steps["arvore"]
        grupos_treino = sorted(grupos_serie.loc[~mascara_teste].unique().tolist())
        metricas.update(
            {
                "rodada": numero,
                "grupo_teste": grupo_teste,
                "grupos_treino": ";".join(grupos_treino),
                "amostras_treino": int(len(treino)),
                "profundidade_arvore": int(arvore.get_depth()),
                "folhas_arvore": int(arvore.get_n_leaves()),
            }
        )
        linhas_metricas.append(metricas)

        pasta_rodada = saida / f"rodada_{numero:02d}_{grupo_teste}"
        base.salvar_resultados(
            pasta_rodada,
            modelo,
            features,
            teste,
            y_teste,
            probabilidades,
            previsoes,
            metricas,
            args.threshold,
        )
        previsoes_fora_amostra.append(
            montar_previsoes_auditoria(
                teste, y_teste, probabilidades, previsoes, grupo_teste
            )
        )
        importancias.append(
            pd.DataFrame(
                {
                    "rodada": numero,
                    "grupo_teste": grupo_teste,
                    "feature": features,
                    "importancia": arvore.feature_importances_,
                }
            )
        )
        print(
            f"[{numero}/{len(grupos)}] teste={grupo_teste}: "
            f"TN={metricas['tn']} FP={metricas['fp']} "
            f"FN={metricas['fn']} TP={metricas['tp']}"
        )

    metricas_df = pd.DataFrame(linhas_metricas)
    metricas_df.drop(columns=["matriz_confusao"]).to_csv(
        saida / "metricas_por_partida.csv", index=False
    )
    previsoes_df = pd.concat(previsoes_fora_amostra, ignore_index=True)
    previsoes_df.to_csv(saida / "previsoes_fora_da_amostra.csv", index=False)
    pd.concat(importancias, ignore_index=True).to_csv(
        saida / "importancia_features_por_rodada.csv", index=False
    )

    y_total = previsoes_df["rotulo_real"].astype(int)
    pred_total = previsoes_df["previsao_modelo"].astype(int).to_numpy()
    prob_total = previsoes_df["probabilidade_hacker"].astype(float).to_numpy()
    acumuladas = dividir_metricas(y_total, pred_total, prob_total)
    pd.DataFrame(
        [[acumuladas["tn"], acumuladas["fp"]], [acumuladas["fn"], acumuladas["tp"]]],
        index=["real_normal", "real_hacker"],
        columns=["previsto_normal", "previsto_hacker"],
    ).to_csv(saida / "matriz_confusao_acumulada.csv")

    colunas_resumo = [
        "accuracy",
        "balanced_accuracy",
        "precision_hacker",
        "recall_hacker",
        "f1_hacker",
        "false_positive_rate",
        "roc_auc",
    ]
    consolidado = {
        "metodo": "Leave-One-Group-Out",
        "grupo_validacao": args.agrupar_por,
        "grupos": grupos,
        "rodadas": len(grupos),
        "amostras_total": int(len(dados)),
        "features": features,
        "arma_usada_como_feature": False,
        "metricas_fora_da_amostra_acumuladas": acumuladas,
        "media_e_desvio_entre_rodadas": {
            coluna: media_desvio(metricas_df, coluna) for coluna in colunas_resumo
        },
        "observacao": (
            "Metricas ausentes em uma rodada significam que a partida de teste "
            "nao possuia a classe necessaria para calcula-las."
        ),
    }
    (saida / "metricas_consolidadas.json").write_text(
        json.dumps(consolidado, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    modelo_final = criar_modelo(args, len(dados))
    modelo_final.fit(dados[features], dados["rotulo_real"].astype(int))
    salvar_modelo_final(
        saida / "modelo_final",
        modelo_final,
        features,
        args,
        grupos,
        len(dados),
    )

    print("\n=== VALIDACAO CONCLUIDA ===")
    print(
        "Matriz acumulada: "
        f"TN={acumuladas['tn']} FP={acumuladas['fp']} "
        f"FN={acumuladas['fn']} TP={acumuladas['tp']}"
    )
    print(f"F1 hacker acumulado: {acumuladas['f1_hacker']}")
    print(f"Taxa de falso positivo acumulada: {acumuladas['false_positive_rate']}")
    print(f"Resultados: {saida}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa validacao Leave-One-Group-Out da arvore anti-cheat."
    )
    parser.add_argument("--manifesto", required=True, help="Manifesto CSV consolidado.")
    parser.add_argument(
        "--saida",
        default=str(PASTA_PROJETO / "models" / "arvore_logo"),
        help="Pasta para os resultados da validacao.",
    )
    parser.add_argument(
        "--agrupar-por",
        choices=["partida_id", "jogador_global"],
        default="partida_id",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--ccp-alpha", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold deve estar entre 0 e 1")
    if args.max_depth < 1 or args.min_samples_leaf < 1 or args.ccp_alpha < 0:
        parser.error("hiperparametros da arvore precisam ser positivos")
    executar(args)


if __name__ == "__main__":
    main()
