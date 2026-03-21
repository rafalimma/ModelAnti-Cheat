import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURAÇÃO
# ==========================================
FILE_IN = r'D:\DayZServerProfiles\log_normal_2026-03-04_20-00-10.log' 
FILE_OUT = 'datasets/dataset_processado.csv'
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
            # Procuro se a linha contém DATA_LOG, não importa o que tem antes
            if "DATA_LOG" in line:
                # 1. Removemos o prefixo "SCRIPT : " limpando tudo antes do DATA_LOG
                content = line.split("DATA_LOG")[-1] 
                
                # 2. Agora dividimos pelo separador '|'
                parts = [p.strip() for p in content.split('|') if p.strip()]
                
                try:
                    # Mapeamento baseado no seu log:
                    # parts[0] = Tempo | parts[1] = ID | parts[2] = X | parts[3] = Y | parts[4] = Z
                    # parts[5] = DirX  | parts[6] = DirZ | parts[7] = Mirando | parts[8] = Arma
                    
                    data.append({
                        'tempo': int(parts[0]),
                        'player_id': parts[1],
                        'pos_x': float(parts[2]),
                        'pos_y': float(parts[3]),   
                        'pos_z': float(parts[4]),
                        'dir_x': float(parts[5]),
                        'dir_z': float(parts[6]),
                        'mirando': int(parts[7]),
                        'arma': parts[8]
                    })
                except (IndexError, ValueError) as e:
                    continue # Pula linhas incompletas

    if not data:
        print("Ainda não encontrei dados. Verifique se o arquivo não está vazio.")
        return

    df = pd.DataFrame(data)
    print(f"Sucesso! {len(df)} linhas processadas.")

    # --- CÁLCULOS COMPORTAMENTAIS ---
    df['vel_posicao'] = 0.0
    df['vel_rotacao'] = 0.0

    for pid in df['player_id'].unique():
        mask = df['player_id'] == pid
        player_df = df[mask].copy()

        # Velocidade de Movimento
        dist = np.sqrt(player_df['pos_x'].diff()**2 + player_df['pos_z'].diff()**2)
        df.loc[mask, 'vel_posicao'] = (dist / dt).fillna(0.0)

        # Velocidade de Rotação (Aimbot)
        angulos = np.arctan2(player_df['dir_x'], player_df['dir_z'])
        diff_ang = angulos.diff()
        # Ajuste de rotação circular
        diff_ang = np.abs(np.arctan2(np.sin(diff_ang), np.cos(diff_ang)))
        df.loc[mask, 'vel_rotacao'] = (diff_ang / dt).fillna(0.0)

    # SALVAMENTO
    # Para a IA, vamos salvar as colunas que importam para o comportamento
    cols_ia = ['pos_x', 'pos_z', 'pos_y', 'vel_posicao', 'vel_rotacao', 'mirando']
    df[cols_ia].to_csv(output_path, index=False)
    print(f"Dataset salvo em: {output_path}")

preprocess_log(FILE_IN, FILE_OUT, INTERVALO_TEMPO)