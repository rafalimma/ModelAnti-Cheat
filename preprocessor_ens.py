import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURAÇÃO
# ==========================================
FILE_IN = r'log_profiles/script_2026-hacker-scar.log'
FILE_OUT = 'datasets/hacker_scar.csv'
INTERVALO_TEMPO = 1.0 
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
                content = line.split("DATA_LOG")[-1] 
                parts = [p.strip() for p in content.split('|') if p.strip()]   
                
                try:
                    # Mapeamento com o novo campo 'tem_visao' (parts[9]):
                    # 0:Tempo | 1:ID | 2:X | 3:Y | 4:Z | 5:DirX | 6:DirZ | 7:Mirando | 8:Arma | 9:TemVisao
                    data.append({
                        'tempo': int(parts[0]),
                        'player_id': parts[1],
                        'pos_x': float(parts[2]),
                        'pos_y': float(parts[3]),   
                        'pos_z': float(parts[4]),
                        'dir_x': float(parts[5]),
                        'dir_z': float(parts[6]),
                        'mirando': int(parts[7]),
                        'arma': parts[8],
                        'tem_visao': int(parts[9]) if len(parts) > 9 else 1 # Leitura do Raycast do Servidor
                    })
                except (IndexError, ValueError):
                    continue

    if not data:
        print("Ainda não encontrei dados. Verifique se o arquivo não está vazio.")
        return

    df = pd.DataFrame(data)
    print(f"Sucesso! {len(df)} linhas processadas.")

    # --- 1. CÁLCULOS COMPORTAMENTAIS INDIVIDUAIS ---
    df['vel_posicao'] = 0.0
    df['vel_rotacao'] = 0.0
    df['acel_linear'] = 0.0
    df['jitter_mira'] = 0.0
    df['entropia_mov'] = 0.0
    df['eficiencia_trajeto'] = 0.0
    df['var_camera_durante_corrida'] = 0.0

    for pid in df['player_id'].unique():
        mask = df['player_id'] == pid
        player_df = df[mask].copy()

        # Distância e Velocidades
        dist = np.sqrt(player_df['pos_x'].diff()**2 + player_df['pos_z'].diff()**2)
        vel_lin = (dist / dt).fillna(0.0)
        acel_lin = (vel_lin.diff() / dt).fillna(0.0)
        
        # Rotação da Câmera
        angulos = np.arctan2(player_df['dir_x'], player_df['dir_z'])
        diff_ang = np.abs(np.arctan2(np.sin(angulos.diff()), np.cos(angulos.diff())))
        vel_rot = (diff_ang / dt).fillna(0.0)

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

        # Mudar de Olhar/Varredura da Câmera (Uso de 'Alt' / Free Look)
        # Se a câmera está se movendo enquanto o jogador corre, é sinal de humano, não de bot
        var_cam = vel_rot.rolling(window=10, min_periods=1).std().fillna(0.0)

        # Atribuição por máscara
        df.loc[mask, 'vel_posicao'] = vel_lin
        df.loc[mask, 'acel_linear'] = acel_lin
        df.loc[mask, 'vel_rotacao'] = vel_rot
        df.loc[mask, 'jitter_mira'] = jitter
        df.loc[mask, 'entropia_mov'] = entropia
        df.loc[mask, 'eficiencia_trajeto'] = eficiencia
        df.loc[mask, 'var_camera_durante_corrida'] = var_cam

        # Médias de Janela Temporal (Persistência)
        df.loc[mask, 'eficiencia_mean_20s'] = pd.Series(eficiencia, index=player_df.index).rolling(20, min_periods=1).mean().fillna(0.0)
        df.loc[mask, 'entropia_mean_10s'] = pd.Series(entropia, index=player_df.index).rolling(10, min_periods=1).mean().fillna(0.0)

    # --- 2. CRUZAMENTO ESPACIAL (ESP / WALLHACK COM RAYCAST) ---
    df['travado_em_player'] = 180.0
    df['perseguindo_player'] = 0
    df['distancia_alvo'] = 0.0

    for tempo_tique in df['tempo'].unique():
        tique_df = df[df['tempo'] == tempo_tique]
        if len(tique_df) < 2:
            continue

        for idx, player_atual in tique_df.iterrows():
            pid_atual = player_atual['player_id']
            px, pz = player_atual['pos_x'], player_atual['pos_z']
            dx, dz = player_atual['dir_x'], player_atual['dir_z']
            menor_desvio_angulo = 180.0
            dist_alvo_mais_proximo = 0.0
            esta_seguindo = 0

            for idx_inimigo, player_inimigo in tique_df.iterrows():
                if player_inimigo['player_id'] == pid_atual:
                    continue

                ix, iz = player_inimigo['pos_x'], player_inimigo['pos_z']
                vetor_inimigo_x = ix - px
                vetor_inimigo_z = iz - pz
                dist_ate_inimigo = np.sqrt(vetor_inimigo_x**2 + vetor_inimigo_z**2)

                if dist_ate_inimigo > 1000 or dist_ate_inimigo == 0:
                    continue

                v_inimigo_x = vetor_inimigo_x / dist_ate_inimigo
                v_inimigo_z = vetor_inimigo_z / dist_ate_inimigo
                
                dot_product = np.clip((dx * v_inimigo_x) + (dz * v_inimigo_z), -1.0, 1.0)
                angulo_desvio = np.degrees(np.arccos(dot_product))

                if angulo_desvio < menor_desvio_angulo:
                    menor_desvio_angulo = angulo_desvio
                    dist_alvo_mais_proximo = dist_ate_inimigo
                
                # Perseguição (B-lining)
                if player_atual['vel_posicao'] > 4.0 and angulo_desvio < 10.0:
                    # Só considera perseguição suspeita se ele NÃO estiver variando a câmera (sem Free Look)
                    if player_atual['eficiencia_trajeto'] > 0.92 and player_atual['var_camera_durante_corrida'] < 0.05:
                        esta_seguindo = 1

            df.at[idx, 'travado_em_player'] = menor_desvio_angulo
            df.at[idx, 'distancia_alvo'] = dist_alvo_mais_proximo
            df.at[idx, 'perseguindo_player'] = esta_seguindo

    # Menor ângulo mantido nos últimos 5s
    for pid in df['player_id'].unique():
        mask = df['player_id'] == pid
        df.loc[mask, 'travado_min_5s'] = df.loc[mask, 'travado_em_player'].rolling(5, min_periods=1).min()

    # --- 3. REGRAS E FLAGS DE TREINAMENTO (AVANÇADAS) ---
    df['alerta_speedhack'] = (df['vel_posicao'] > 9.0).astype(int)
    df['alerta_aimbot'] = (df['vel_rotacao'] > 15.0).astype(int)
    df['alerta_lockon'] = ((df['jitter_mira'] < 0.001) & (df['vel_posicao'] > 0.0)).astype(int)
    
    # NOVO: Alerta ESP que considera a OCLUSÃO (Raycast == 0) e Distância (> 150m)
    # Ativa se está mirando com baixo ângulo de desvio (<10°), SEM VISÃO DIRETA (através da parede/montanha) e a mais de 150m
    df['esp_oculto_longa_distancia'] = (
        (df['travado_em_player'] < 10.0) & 
        (df['tem_visao'] == 0) & 
        (df['distancia_alvo'] > 150.0)
    ).astype(int)

    # --- 4. EXPORTAÇÃO COMPLETA PARA A IA ---
    cols_ia = [
        'vel_posicao', 'acel_linear', 'vel_rotacao', 'jitter_mira', 
        'entropia_mov', 'eficiencia_trajeto', 'var_camera_durante_corrida',
        'eficiencia_mean_20s', 'entropia_mean_10s', 'travado_min_5s',
        'travado_em_player', 'distancia_alvo', 'tem_visao', 
        'perseguindo_player', 'mirando',
        'alerta_speedhack', 'alerta_aimbot', 'alerta_lockon', 'esp_oculto_longa_distancia'
    ]

    df_final = df[cols_ia].replace([np.inf, -np.inf], np.nan).dropna()
    df_final.to_csv(output_path, index=False)
    print(f"Dataset processado e salvo com sucesso em: {output_path}")

preprocess_log(FILE_IN, FILE_OUT, INTERVALO_TEMPO)