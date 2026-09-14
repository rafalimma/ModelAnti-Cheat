import pandas as pd
import numpy as np
import os
import argparse
# senha do server 098765
# ==========================================
# CONFIGURAÇÃO
# ==========================================
FILE_IN = r'log_profiles/script_2026-09-04_23-42-20.log'
FILE_OUT = 'datasets/partida_1acao_hacker_raw.csv'
INTERVALO_TEMPO = 1.0 
ANGULO_MIRA_DEG = 5.0
ANGULO_PRECISAO_ESP_DEG = 1.0
DISTANCIA_ESP_MIN = 300.0
DURACAO_ESP_SEG = 3.0
DURACAO_PRECISAO_ESP_SEG = 3.0
JANELA_REPETICAO_ESP_SEG = 120.0
MIN_EPISODIOS_ESP_FORTE = 2
DURACAO_MOVIMENTO_SEG = 4.0
JANELA_ANALISE_SEG = 10.0
JANELA_PONTOS_SEG = 60.0
LIMIAR_REVISAO = 3
LIMIAR_BANIMENTO = 6

# Somente estas colunas devem entrar no X do modelo. Pontuação, nível e decisão
# ficam fora para impedir vazamento da resposta criada pelo motor de regras.
FEATURES_IA_RECOMENDADAS = [
    'vel_posicao', 'acel_linear', 'vel_rotacao', 'jitter_mira',
    'entropia_mov', 'eficiencia_trajeto', 'var_camera_durante_corrida',
    'eficiencia_mean_20s', 'entropia_mean_10s', 'travado_min_5s',
    'travado_em_player', 'distancia_alvo', 'tem_visao',
    'alvo_encontrado', 'alvo_na_mira', 'mirando', 'mirando_ads',
    'mirando_alvo', 'alvo_oculto_na_mira', 'mirando_alvo_oculto',
    'alvo_oculto_300m_na_mira', 'mirando_alvo_oculto_300m',
    'mirando_ads_alvo_oculto_300m', 'alinhamento_preciso_oculto_300m',
    'duracao_mirando_alvo_s', 'duracao_rastreando_oculto_s',
    'duracao_mirando_oculto_s', 'duracao_observando_oculto_300m_s',
    'duracao_mirando_oculto_300m_s', 'duracao_ads_oculto_300m_s',
    'duracao_precisao_oculta_300m_s', 'contagem_episodios_esp_120s',
    'contagem_episodios_ads_120s', 'taxa_alvo_na_mira_10s',
    'taxa_alvo_oculto_10s', 'taxa_mirando_alvo_10s',
    'taxa_mira_oculta_10s', 'taxa_visao_direta_10s',
    'angulo_medio_alvo_10s', 'angulo_minimo_alvo_10s',
    'distancia_media_alvo_10s', 'vel_rotacao_media_10s',
    'vel_rotacao_std_10s', 'jitter_medio_10s',
    'persistencia_oculta_max_10s', 'persistencia_mira_oculta_max_10s',
    'taxa_alvo_oculto_300m_10s', 'taxa_mira_oculta_300m_10s',
    'taxa_ads_oculta_300m_10s', 'taxa_precisao_oculta_300m_10s',
    'persistencia_oculta_300m_max_10s', 'persistencia_mira_oculta_300m_max_10s',
    'perseguindo_player', 'duracao_perseguicao_oculta_s',
    'duracao_speedhack_s'
]

FEATURES_JANELA_IA_RECOMENDADAS = [
    'vel_posicao_media', 'vel_posicao_max', 'aceleracao_media',
    'aceleracao_max', 'vel_rotacao_media', 'vel_rotacao_max',
    'vel_rotacao_std', 'jitter_medio', 'entropia_media',
    'eficiencia_trajeto_media', 'variacao_camera_media',
    'angulo_alvo_medio', 'angulo_alvo_min', 'distancia_alvo_media',
    'distancia_alvo_max', 'taxa_alvo_encontrado', 'taxa_alvo_na_mira',
    'taxa_visao_direta', 'taxa_alvo_oculto', 'taxa_mirando_alvo',
    'taxa_mira_oculta', 'duracao_rastreando_oculto_max',
    'duracao_mirando_oculto_max', 'duracao_perseguicao_oculta_max',
    'taxa_alvo_oculto_300m', 'taxa_mira_oculta_300m',
    'duracao_observando_oculto_300m_max', 'duracao_mirando_oculto_300m_max',
    'taxa_ads_oculta_300m', 'taxa_precisao_oculta_300m',
    'duracao_ads_oculto_300m_max', 'duracao_precisao_oculta_300m_max',
    'contagem_episodios_esp_120s_max', 'contagem_episodios_ads_120s_max',
    'duracao_speedhack_max'
]

# ==========================================

def preprocess_log(input_path, output_path, dt):
    print(f"Lendo log: {input_path}...")
    data = []

    if not os.path.exists(input_path):
        print("Erro: Arquivo não encontrado.")
        return

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "DATA_LOG" in line:
                content = line.split("DATA_LOG", 1)[1]
                # Não remova campos vazios: blockerType pode vir vazio e deslocaria
                # todos os diagnósticos posteriores do schema.
                parts = [p.strip() for p in content.split('|')]
                if parts and parts[0] == '':
                    parts = parts[1:]
                
                try:
                    # 0..11 mantêm compatibilidade com o log antigo. O schema novo
                    # acrescenta DirY, presença/tipo do alvo e flags calculadas no servidor.
                    distancia_alvo = float(parts[11]) if len(parts) > 11 else 0.0
                    angulo_alvo = float(parts[10]) if len(parts) > 10 else 180.0
                    alvo_encontrado = int(parts[13]) if len(parts) > 13 else int(distancia_alvo > 0.0)
                    alvo_na_mira = int(parts[15]) if len(parts) > 15 else int(alvo_encontrado == 1 and angulo_alvo <= ANGULO_MIRA_DEG)
                    data.append({
                        'tempo': int(parts[0]),
                        'player_id': parts[1],
                        'pos_x': float(parts[2]),
                        'pos_y': float(parts[3]),   
                        'pos_z': float(parts[4]),
                        'dir_x': float(parts[5]),
                        'dir_y': float(parts[12]) if len(parts) > 12 else 0.0,
                        'dir_z': float(parts[6]),
                        'mirando': int(parts[7]),
                        'mirando_ads': int(parts[18]) if len(parts) > 18 and parts[18] else 0,
                        'arma': parts[8],
                        'tem_visao': int(parts[9]) if len(parts) > 9 else 1,
                        'travado_em_player': angulo_alvo,
                        'distancia_alvo': distancia_alvo,
                        'alvo_encontrado': alvo_encontrado,
                        # O tipo do alvo é intencionalmente ignorado. Dummy e jogador
                        # real são tratados como o mesmo alvo humano pelo dataset.
                        'alvo_na_mira': alvo_na_mira
                    })
                except (IndexError, ValueError):
                    continue

    if not data:
        print("Ainda não encontrei dados. Verifique se o arquivo não está vazio.")
        return

    df = pd.DataFrame(data)
    # Entidades artificiais sem PlayerIdentity eram registradas como observadores
    # pelo init.c antigo. Elas são alvos, não jogadores a serem classificados.
    df = df[df['player_id'].str.lower() != 'unknown'].copy()
    if df.empty:
        print("Nenhum jogador com identidade válida foi encontrado no log.")
        return
    df = df.sort_values(['player_id', 'tempo'], kind='stable').reset_index(drop=True)
    df['jogador_sessao'] = pd.factorize(df['player_id'], sort=False)[0] + 1
    df['tempo_run_s'] = (
        df['tempo'] - df.groupby('player_id', sort=False)['tempo'].transform('min')
    ) / 1000.0
    df['janela_analise_10s'] = np.floor(df['tempo_run_s'] / JANELA_ANALISE_SEG).astype(int)
    print(f"Sucesso! {len(df)} linhas processadas.")

    # --- 1. CÁLCULOS COMPORTAMENTAIS INDIVIDUAIS ---
    df['vel_posicao'] = 0.0
    df['vel_rotacao'] = 0.0
    df['acel_linear'] = 0.0
    df['jitter_mira'] = 0.0
    df['entropia_mov'] = 0.0
    df['eficiencia_trajeto'] = 0.0
    df['var_camera_durante_corrida'] = 0.0
    df['delta_t'] = float(dt)

    for pid in df['player_id'].unique():
        mask = df['player_id'] == pid
        player_df = df[mask].copy()

        # Usa o relógio real do servidor (milissegundos), evitando assumir tick perfeito.
        delta_t = (player_df['tempo'].diff() / 1000.0)
        delta_t = delta_t.where((delta_t > 0.0) & (delta_t <= 10.0), dt).fillna(dt)

        # Distância e velocidades 3D.
        dist = np.sqrt(player_df['pos_x'].diff()**2 + player_df['pos_y'].diff()**2 + player_df['pos_z'].diff()**2)
        vel_lin = (dist / delta_t).fillna(0.0)
        acel_lin = (vel_lin.diff() / delta_t).fillna(0.0)
        
        # Rotação da Câmera
        dirs = player_df[['dir_x', 'dir_y', 'dir_z']].to_numpy(dtype=float)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = np.divide(dirs, norms, out=np.zeros_like(dirs), where=norms > 0.0)
        dots = np.sum(dirs[1:] * dirs[:-1], axis=1)
        diff_ang = pd.Series(np.r_[0.0, np.arccos(np.clip(dots, -1.0, 1.0))], index=player_df.index)
        vel_rot = (diff_ang / delta_t).fillna(0.0)

        # Jitter e Entropia
        jitter = vel_rot.diff().abs().fillna(0.0)
        entropia = vel_rot.rolling(window=5, min_periods=1).std().fillna(0.0)

        # Eficiência de Trajetória
        janela = 20
        dist_acumulada = dist.rolling(window=janela, min_periods=1).sum()
        dx_20 = player_df['pos_x'] - player_df['pos_x'].shift(janela)
        dz_20 = player_df['pos_z'] - player_df['pos_z'].shift(janela)
        dist_reta_20 = np.sqrt(dx_20**2 + dz_20**2)
        
        eficiencia = np.where(dist_acumulada > 0.1, dist_reta_20 / dist_acumulada, 0.0)
        eficiencia = np.nan_to_num(eficiencia, nan=0.0)

        # Varredura da Câmera (Free Look / Alt)
        var_cam = vel_rot.rolling(window=10, min_periods=1).std().fillna(0.0)

        # Atribuição por máscara
        df.loc[mask, 'vel_posicao'] = vel_lin
        df.loc[mask, 'acel_linear'] = acel_lin
        df.loc[mask, 'vel_rotacao'] = vel_rot
        df.loc[mask, 'jitter_mira'] = jitter
        df.loc[mask, 'entropia_mov'] = entropia
        df.loc[mask, 'eficiencia_trajeto'] = eficiencia
        df.loc[mask, 'var_camera_durante_corrida'] = var_cam
        df.loc[mask, 'delta_t'] = delta_t

        # Médias de Janela Temporal (Persistência)
        df.loc[mask, 'eficiencia_mean_20s'] = pd.Series(eficiencia, index=player_df.index).rolling(20, min_periods=1).mean().fillna(0.0)
        df.loc[mask, 'entropia_mean_10s'] = pd.Series(entropia, index=player_df.index).rolling(10, min_periods=1).mean().fillna(0.0)

        # Menor ângulo mantido nos últimos 5s (calculado individualmente por player)
        df.loc[mask, 'travado_min_5s'] = df.loc[mask, 'travado_em_player'].rolling(5, min_periods=1).min()

    # --- 2. ESTADOS GERAIS DO ALVO (DUMMY E PLAYER REAL) ---
    # O dummy é apenas um PlayerBase de teste. Nenhuma feature abaixo revela seu tipo.
    df['mirando'] = ((df['mirando'] == 1) | (df['mirando_ads'] == 1)).astype(int)
    df['mirando_alvo'] = (
        (df['alvo_encontrado'] == 1) &
        (df['alvo_na_mira'] == 1) &
        (df['mirando'] == 1)
    ).astype(int)
    df['alvo_oculto_na_mira'] = (
        (df['alvo_encontrado'] == 1) &
        (df['alvo_na_mira'] == 1) &
        (df['tem_visao'] == 0)
    ).astype(int)
    df['mirando_alvo_oculto'] = (
        (df['alvo_oculto_na_mira'] == 1) &
        (df['mirando'] == 1)
    ).astype(int)
    # Estado neutro usado para ESP/X-ray: o alvo está alinhado, oculto e além
    # da distância mínima configurada. Olhar e mirar armado são preservados
    # separadamente para que a IA aprenda intensidades diferentes.
    df['alvo_oculto_300m_na_mira'] = (
        (df['alvo_oculto_na_mira'] == 1) &
        (df['distancia_alvo'] > DISTANCIA_ESP_MIN)
    ).astype(int)
    df['mirando_alvo_oculto_300m'] = (
        (df['alvo_oculto_300m_na_mira'] == 1) &
        (df['mirando'] == 1)
    ).astype(int)
    df['mirando_ads_alvo_oculto_300m'] = (
        (df['alvo_oculto_300m_na_mira'] == 1) &
        (df['mirando_ads'] == 1)
    ).astype(int)
    df['alinhamento_preciso_oculto_300m'] = (
        (df['mirando_alvo_oculto_300m'] == 1) &
        (df['travado_em_player'] <= ANGULO_PRECISAO_ESP_DEG)
    ).astype(int)

    def duracao_consecutiva(condicao):
        """Soma o tempo real apenas enquanto a condição permanece contínua."""
        resultado = pd.Series(0.0, index=df.index)
        for _, indices in df.groupby('player_id', sort=False).groups.items():
            idx = list(indices)
            ativa = condicao.loc[idx].astype(bool)
            grupos = (~ativa).cumsum()
            # A primeira amostra da sequência começa em zero. Só acumulamos o
            # delta real quando a amostra anterior também estava ativa.
            anterior_ativa = ativa.shift(1, fill_value=False)
            incremento = df.loc[idx, 'delta_t'].where(ativa & anterior_ativa, 0.0)
            duracao = incremento.groupby(grupos).cumsum()
            resultado.loc[idx] = duracao.to_numpy()
        return resultado

    def evento_novo_bloco(duracao, tamanho_bloco_seg):
        """Gera um evento a cada novo bloco completo de tempo real."""
        blocos = np.floor(duracao / tamanho_bloco_seg).astype(int)
        anterior = blocos.groupby(df['player_id'], sort=False).shift(1).fillna(0).astype(int)
        return (blocos > anterior).astype(int)

    def cruzou_limiar(duracao, limiar_seg):
        """Marca uma vez cada episódio que alcança o tempo mínimo configurado."""
        anterior = duracao.groupby(df['player_id'], sort=False).shift(1).fillna(0.0)
        return ((duracao >= limiar_seg) & (anterior < limiar_seg)).astype(int)

    janela_analise = max(1, int(round(JANELA_ANALISE_SEG / max(float(dt), 0.1))))
    janela_pontos = max(1, int(round(JANELA_PONTOS_SEG / max(float(dt), 0.1))))

    df['duracao_mirando_alvo_s'] = duracao_consecutiva(df['mirando_alvo'] == 1)
    df['duracao_rastreando_oculto_s'] = duracao_consecutiva(df['alvo_oculto_na_mira'] == 1)
    df['duracao_mirando_oculto_s'] = duracao_consecutiva(df['mirando_alvo_oculto'] == 1)
    df['duracao_observando_oculto_300m_s'] = duracao_consecutiva(
        df['alvo_oculto_300m_na_mira'] == 1
    )
    df['duracao_mirando_oculto_300m_s'] = duracao_consecutiva(
        df['mirando_alvo_oculto_300m'] == 1
    )
    df['duracao_ads_oculto_300m_s'] = duracao_consecutiva(
        df['mirando_ads_alvo_oculto_300m'] == 1
    )
    df['duracao_precisao_oculta_300m_s'] = duracao_consecutiva(
        df['alinhamento_preciso_oculto_300m'] == 1
    )
    df['tracking_oculto_suspeito'] = (
        df['duracao_observando_oculto_300m_s'] >= DURACAO_ESP_SEG
    ).astype(int)
    df['mira_armada_oculta_suspeita'] = (
        df['duracao_mirando_oculto_300m_s'] >= DURACAO_ESP_SEG
    ).astype(int)

    # Features temporais neutras para a IA. Elas descrevem o comportamento na
    # faixa recente, mas não contêm a classificação produzida pelas regras.
    def rolling_por_jogador(coluna, operacao, min_periods=1):
        return df.groupby('player_id', sort=False)[coluna].transform(
            lambda s: getattr(s.rolling(janela_analise, min_periods=min_periods), operacao)()
        )

    df['taxa_alvo_na_mira_10s'] = rolling_por_jogador('alvo_na_mira', 'mean')
    df['taxa_alvo_oculto_10s'] = rolling_por_jogador('alvo_oculto_na_mira', 'mean')
    df['taxa_mirando_alvo_10s'] = rolling_por_jogador('mirando_alvo', 'mean')
    df['taxa_mira_oculta_10s'] = rolling_por_jogador('mirando_alvo_oculto', 'mean')
    df['taxa_alvo_oculto_300m_10s'] = rolling_por_jogador(
        'alvo_oculto_300m_na_mira', 'mean'
    )
    df['taxa_mira_oculta_300m_10s'] = rolling_por_jogador(
        'mirando_alvo_oculto_300m', 'mean'
    )
    df['taxa_ads_oculta_300m_10s'] = rolling_por_jogador(
        'mirando_ads_alvo_oculto_300m', 'mean'
    )
    df['taxa_precisao_oculta_300m_10s'] = rolling_por_jogador(
        'alinhamento_preciso_oculto_300m', 'mean'
    )
    df['vel_rotacao_media_10s'] = rolling_por_jogador('vel_rotacao', 'mean')
    df['vel_rotacao_std_10s'] = rolling_por_jogador('vel_rotacao', 'std').fillna(0.0)
    df['jitter_medio_10s'] = rolling_por_jogador('jitter_mira', 'mean')
    df['persistencia_oculta_max_10s'] = rolling_por_jogador(
        'duracao_rastreando_oculto_s', 'max'
    )
    df['persistencia_mira_oculta_max_10s'] = rolling_por_jogador(
        'duracao_mirando_oculto_s', 'max'
    )
    df['persistencia_oculta_300m_max_10s'] = rolling_por_jogador(
        'duracao_observando_oculto_300m_s', 'max'
    )
    df['persistencia_mira_oculta_300m_max_10s'] = rolling_por_jogador(
        'duracao_mirando_oculto_300m_s', 'max'
    )

    alvo_valido = df['alvo_encontrado'].astype(float)
    visao_valida = ((df['alvo_encontrado'] == 1) & (df['tem_visao'] == 1)).astype(float)
    soma_alvos = alvo_valido.groupby(df['player_id'], sort=False).transform(
        lambda s: s.rolling(janela_analise, min_periods=1).sum()
    )
    soma_visao = visao_valida.groupby(df['player_id'], sort=False).transform(
        lambda s: s.rolling(janela_analise, min_periods=1).sum()
    )
    df['taxa_visao_direta_10s'] = np.divide(
        soma_visao, soma_alvos,
        out=np.zeros(len(df), dtype=float),
        where=soma_alvos.to_numpy() > 0.0
    )

    angulo_valido = df['travado_em_player'].where(df['alvo_encontrado'] == 1)
    distancia_valida = df['distancia_alvo'].where(df['alvo_encontrado'] == 1)
    df['angulo_medio_alvo_10s'] = angulo_valido.groupby(
        df['player_id'], sort=False
    ).transform(lambda s: s.rolling(janela_analise, min_periods=1).mean()).fillna(180.0)
    df['angulo_minimo_alvo_10s'] = angulo_valido.groupby(
        df['player_id'], sort=False
    ).transform(lambda s: s.rolling(janela_analise, min_periods=1).min()).fillna(180.0)
    df['distancia_media_alvo_10s'] = distancia_valida.groupby(
        df['player_id'], sort=False
    ).transform(lambda s: s.rolling(janela_analise, min_periods=1).mean()).fillna(0.0)

    # Perseguição Direta (B-lining)
    df['perseguindo_player'] = np.where(
        (df['vel_posicao'] > 4.0) &
        (df['travado_em_player'] < 10.0) &
        (df['tem_visao'] == 0) &
        (df['distancia_alvo'] > DISTANCIA_ESP_MIN) &
        (df['eficiencia_trajeto'] > 0.92) &
        (df['var_camera_durante_corrida'] < 0.05),
        1, 0
    )

    # --- 3. EVIDÊNCIAS E PONTUAÇÃO EXPLICÁVEL ---
    # Speedhack exige persistência para não punir um salto isolado causado por lag.
    df['duracao_speedhack_s'] = duracao_consecutiva(df['vel_posicao'] > 9.0)
    df['alerta_speedhack'] = (
        df['duracao_speedhack_s'] >= DURACAO_MOVIMENTO_SEG
    ).astype(int)

    # Com telemetria de 1 Hz, 720 graus/s seria inalcançável matematicamente.
    # Um snap acima de 120 graus/s só pontua se terminar alinhado e com arma/ADS.
    df['alerta_aimbot'] = (
        (df['vel_rotacao'] > np.deg2rad(120.0)) &
        (df['mirando_alvo'] == 1)
    ).astype(int)
    df['alerta_lockon'] = (
        (df['duracao_mirando_alvo_s'] >= DURACAO_MOVIMENTO_SEG) &
        (df['jitter_mira'] < 0.01)
    ).astype(int)

    # O evento primário de ESP/X-ray exige arma/mira ativa. Apenas olhar para um
    # alvo oculto continua disponível como telemetria e gera no máximo suspeita leve.
    df['esp_oculto_longa_distancia'] = (
        df['duracao_mirando_oculto_300m_s'] >= DURACAO_ESP_SEG
    ).astype(int)

    df['duracao_perseguicao_oculta_s'] = duracao_consecutiva(df['perseguindo_player'] == 1)

    # Observação sem arma marca somente o início de cada episódio. A mira oculta
    # armada pode acumular evidência conforme permanece contínua.
    evento_observacao_leve = cruzou_limiar(
        df['duracao_observando_oculto_300m_s'], DURACAO_ESP_SEG
    )
    evento_esp_primario = evento_novo_bloco(
        df['duracao_mirando_oculto_300m_s'], DURACAO_ESP_SEG
    )
    episodio_esp_novo = cruzou_limiar(
        df['duracao_mirando_oculto_300m_s'], DURACAO_ESP_SEG
    )
    episodio_ads_novo = cruzou_limiar(
        df['duracao_ads_oculto_300m_s'], DURACAO_ESP_SEG
    )
    evento_precisao_oculta = cruzou_limiar(
        df['duracao_precisao_oculta_300m_s'], DURACAO_PRECISAO_ESP_SEG
    )
    evento_perseguicao = evento_novo_bloco(
        df['duracao_perseguicao_oculta_s'], DURACAO_ESP_SEG
    )
    evento_speedhack = evento_novo_bloco(
        df['duracao_speedhack_s'], DURACAO_MOVIMENTO_SEG
    )

    janela_repeticao = max(
        1, int(round(JANELA_REPETICAO_ESP_SEG / max(float(dt), 0.1)))
    )
    df['contagem_episodios_esp_120s'] = episodio_esp_novo.groupby(
        df['player_id'], sort=False
    ).transform(lambda s: s.rolling(janela_repeticao, min_periods=1).sum()).astype(int)
    df['contagem_episodios_ads_120s'] = episodio_ads_novo.groupby(
        df['player_id'], sort=False
    ).transform(lambda s: s.rolling(janela_repeticao, min_periods=1).sum()).astype(int)

    df['evidencia_esp_repetido'] = (
        df['contagem_episodios_esp_120s'] >= MIN_EPISODIOS_ESP_FORTE
    ).astype(int)
    df['evidencia_ads_oculto_repetido'] = (
        df['contagem_episodios_ads_120s'] >= MIN_EPISODIOS_ESP_FORTE
    ).astype(int)
    df['evidencia_precisao_oculta'] = (
        df['duracao_precisao_oculta_300m_s'] >= DURACAO_PRECISAO_ESP_SEG
    ).astype(int)

    # Perseguição é evidência secundária: só pontua se houve um ESP primário
    # armado nos últimos 60 segundos. Isoladamente, nunca cria suspeita de hack.
    esp_primario_recente = episodio_esp_novo.groupby(
        df['player_id'], sort=False
    ).transform(lambda s: s.rolling(janela_pontos, min_periods=1).max()).astype(int)
    evento_perseguicao_corrobora = (evento_perseguicao & esp_primario_recente).astype(int)

    def inicio_evidencia(coluna):
        anterior = coluna.groupby(df['player_id'], sort=False).shift(1).fillna(0).astype(int)
        return ((coluna == 1) & (anterior == 0)).astype(int)

    evento_esp_repetido = inicio_evidencia(df['evidencia_esp_repetido'])
    evento_ads_repetido = inicio_evidencia(df['evidencia_ads_oculto_repetido'])

    # A observação desarmada vale apenas um ponto por episódio. Mira oculta armada,
    # precisão extrema e repetição têm peso maior; perseguição apenas corrobora.
    df['pontos_evento'] = (
        evento_observacao_leve +
        (2 * evento_esp_primario) +
        evento_perseguicao_corrobora +
        (2 * evento_precisao_oculta) +
        (2 * evento_esp_repetido) +
        (2 * evento_ads_repetido) +
        df['alerta_aimbot'] +
        (2 * evento_speedhack)
    ).astype(int)
    df['pontos_suspeita_60s'] = (
        df.groupby('player_id', sort=False)['pontos_evento']
          .transform(lambda s: s.rolling(janela_pontos, min_periods=1).sum())
          .astype(int)
    )

    tipos_evento = pd.DataFrame({
        'esp_primario': episodio_esp_novo,
        'esp_repetido': evento_esp_repetido,
        'ads_oculto_repetido': evento_ads_repetido,
        'precisao_oculta': evento_precisao_oculta,
        'aimbot': df['alerta_aimbot'],
        'speedhack': evento_speedhack
    })
    df['tipos_evidencia_60s'] = 0
    for coluna in tipos_evento.columns:
        recente = tipos_evento[coluna].groupby(df['player_id'], sort=False).transform(
            lambda s: s.rolling(janela_pontos, min_periods=1).max()
        )
        df['tipos_evidencia_60s'] += recente.astype(int)

    df['candidato_revisao'] = (
        df['pontos_suspeita_60s'] >= LIMIAR_REVISAO
    ).astype(int)

    evidencia_forte_esp = (
        (df['evidencia_ads_oculto_repetido'] == 1) |
        (df['evidencia_precisao_oculta'] == 1) |
        (df['evidencia_esp_repetido'] == 1)
    )
    df['esp_alto_risco'] = (
        (esp_primario_recente == 1) &
        evidencia_forte_esp &
        (df['pontos_suspeita_60s'] >= LIMIAR_BANIMENTO)
    ).astype(int)

    # O nome é mantido por compatibilidade, mas significa revisão prioritária.
    # Não deve provocar banimento automático no servidor.
    df['candidato_banimento'] = (
        (df['esp_alto_risco'] == 1) |
        (df['duracao_speedhack_s'] >= 10.0)
    ).astype(int)
    df['nivel_suspeita'] = np.select(
        [
            df['candidato_banimento'] == 1,
            df['candidato_revisao'] == 1,
            df['pontos_suspeita_60s'] >= 1
        ],
        [3, 2, 1],
        default=0
    ).astype(int)

    # Alias geral mantido por compatibilidade com análises anteriores.
    df['wallhack_suspeito'] = df['esp_oculto_longa_distancia']

    # --- 4. EXPORTAÇÃO: FEATURES NEUTRAS + LEITURA HUMANA ---
    # As três primeiras colunas são metadados para localizar a faixa da run e não
    # devem ser usadas como features pelo modelo.
    metadados = ['jogador_sessao', 'tempo_run_s', 'janela_analise_10s']
    colunas_analise = [
        'tracking_oculto_suspeito', 'mira_armada_oculta_suspeita',
        'wallhack_suspeito', 'alerta_speedhack', 'alerta_aimbot',
        'alerta_lockon', 'esp_oculto_longa_distancia',
        'evidencia_esp_repetido', 'evidencia_ads_oculto_repetido',
        'evidencia_precisao_oculta', 'esp_alto_risco', 'pontos_evento',
        'pontos_suspeita_60s', 'tipos_evidencia_60s', 'nivel_suspeita',
        'candidato_revisao', 'candidato_banimento'
    ]

    df_final = df[metadados + FEATURES_IA_RECOMENDADAS + colunas_analise]
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Dataset processado e salvo com sucesso em: {output_path}")

    # Uma segunda saída agrega a telemetria em faixas fixas de 10 segundos. Esse
    # formato reduz o ruído de linhas isoladas e facilita inspeção e treinamento.
    df['_visao_valida'] = np.where(
        df['alvo_encontrado'] == 1,
        (df['tem_visao'] == 1).astype(float),
        np.nan
    )
    df['_angulo_valido'] = df['travado_em_player'].where(df['alvo_encontrado'] == 1)
    df['_distancia_valida'] = df['distancia_alvo'].where(df['alvo_encontrado'] == 1)

    janelas = df.groupby(
        ['jogador_sessao', 'janela_analise_10s'], sort=False
    ).agg(
        inicio_janela_s=('tempo_run_s', 'min'),
        fim_janela_s=('tempo_run_s', 'max'),
        amostras=('tempo_run_s', 'size'),
        vel_posicao_media=('vel_posicao', 'mean'),
        vel_posicao_max=('vel_posicao', 'max'),
        aceleracao_media=('acel_linear', 'mean'),
        aceleracao_max=('acel_linear', 'max'),
        vel_rotacao_media=('vel_rotacao', 'mean'),
        vel_rotacao_max=('vel_rotacao', 'max'),
        vel_rotacao_std=('vel_rotacao', 'std'),
        jitter_medio=('jitter_mira', 'mean'),
        entropia_media=('entropia_mov', 'mean'),
        eficiencia_trajeto_media=('eficiencia_trajeto', 'mean'),
        variacao_camera_media=('var_camera_durante_corrida', 'mean'),
        angulo_alvo_medio=('_angulo_valido', 'mean'),
        angulo_alvo_min=('_angulo_valido', 'min'),
        distancia_alvo_media=('_distancia_valida', 'mean'),
        distancia_alvo_max=('_distancia_valida', 'max'),
        taxa_alvo_encontrado=('alvo_encontrado', 'mean'),
        taxa_alvo_na_mira=('alvo_na_mira', 'mean'),
        taxa_visao_direta=('_visao_valida', 'mean'),
        taxa_alvo_oculto=('alvo_oculto_na_mira', 'mean'),
        taxa_mirando_alvo=('mirando_alvo', 'mean'),
        taxa_mira_oculta=('mirando_alvo_oculto', 'mean'),
        taxa_alvo_oculto_300m=('alvo_oculto_300m_na_mira', 'mean'),
        taxa_mira_oculta_300m=('mirando_alvo_oculto_300m', 'mean'),
        taxa_ads_oculta_300m=('mirando_ads_alvo_oculto_300m', 'mean'),
        taxa_precisao_oculta_300m=('alinhamento_preciso_oculto_300m', 'mean'),
        duracao_rastreando_oculto_max=('duracao_rastreando_oculto_s', 'max'),
        duracao_mirando_oculto_max=('duracao_mirando_oculto_s', 'max'),
        duracao_observando_oculto_300m_max=('duracao_observando_oculto_300m_s', 'max'),
        duracao_mirando_oculto_300m_max=('duracao_mirando_oculto_300m_s', 'max'),
        duracao_ads_oculto_300m_max=('duracao_ads_oculto_300m_s', 'max'),
        duracao_precisao_oculta_300m_max=('duracao_precisao_oculta_300m_s', 'max'),
        contagem_episodios_esp_120s_max=('contagem_episodios_esp_120s', 'max'),
        contagem_episodios_ads_120s_max=('contagem_episodios_ads_120s', 'max'),
        duracao_perseguicao_oculta_max=('duracao_perseguicao_oculta_s', 'max'),
        duracao_speedhack_max=('duracao_speedhack_s', 'max'),
        pontos_evento_total=('pontos_evento', 'sum'),
        pontos_suspeita_60s_max=('pontos_suspeita_60s', 'max'),
        tipos_evidencia_60s_max=('tipos_evidencia_60s', 'max'),
        nivel_suspeita_max=('nivel_suspeita', 'max'),
        candidato_revisao=('candidato_revisao', 'max'),
        candidato_banimento=('candidato_banimento', 'max')
    ).reset_index()

    janelas['vel_rotacao_std'] = janelas['vel_rotacao_std'].fillna(0.0)
    janelas['angulo_alvo_medio'] = janelas['angulo_alvo_medio'].fillna(180.0)
    janelas['angulo_alvo_min'] = janelas['angulo_alvo_min'].fillna(180.0)
    janelas['distancia_alvo_media'] = janelas['distancia_alvo_media'].fillna(0.0)
    janelas['distancia_alvo_max'] = janelas['distancia_alvo_max'].fillna(0.0)
    janelas['taxa_visao_direta'] = janelas['taxa_visao_direta'].fillna(0.0)
    # Índice explicável das regras, não uma probabilidade produzida pela IA.
    janelas['indice_regra_0a1'] = np.clip(
        janelas['pontos_suspeita_60s_max'] / float(LIMIAR_BANIMENTO), 0.0, 1.0
    )
    janelas['padrao_suspeito'] = janelas['nivel_suspeita_max'].astype(int)

    metadados_janela = [
        'jogador_sessao', 'janela_analise_10s', 'inicio_janela_s',
        'fim_janela_s', 'amostras'
    ]
    analise_janela = [
        'pontos_evento_total', 'pontos_suspeita_60s_max',
        'tipos_evidencia_60s_max', 'nivel_suspeita_max',
        'indice_regra_0a1', 'padrao_suspeito',
        'candidato_revisao', 'candidato_banimento'
    ]
    janelas = janelas[
        metadados_janela + FEATURES_JANELA_IA_RECOMENDADAS + analise_janela
    ]

    raiz_saida, extensao_saida = os.path.splitext(output_path)
    caminho_janelas = f"{raiz_saida}_janelas{extensao_saida or '.csv'}"
    janelas.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_csv(
        caminho_janelas, index=False
    )
    print(f"Resumo por janelas de 10s salvo com sucesso em: {caminho_janelas}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converte o DATA_LOG do DayZ em features gerais de anti-cheat.'
    )
    parser.add_argument('input', nargs='?', default=FILE_IN, help='Arquivo de log do servidor.')
    parser.add_argument('output', nargs='?', default=FILE_OUT, help='CSV de saída.')
    parser.add_argument('--dt', type=float, default=INTERVALO_TEMPO, help='Intervalo padrão entre amostras, em segundos.')
    args = parser.parse_args()
    preprocess_log(args.input, args.output, args.dt)
