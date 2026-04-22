import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURAÇÃO
# ==========================================
FILE_IN = r'D:\DayZServerProfiles\script_2026-04-22_10-13-57.log' 
FILE_OUT = 'datasets/dataset_cooked.csv'
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
    df['acel_linear'] = 0.0
    df['jitter_mira'] = 0.0

    #
    for pid in df['player_id'].unique(): # para cada unico player no dataframe
        # assim a velocidade calculada comparando a posição atual do jogadorA com a posição anterior do próprio jogadorA.
        mask = df['player_id'] == pid
        player_df = df[mask].copy()

        # Velocidade de Movimento
        # Teorema de pitágoras para medir a distancia entre dois tiques do jogo
        dist = np.sqrt(player_df['pos_x'].diff()**2 + player_df['pos_z'].diff()**2)
        vel_lin = (dist / dt).fillna(0.0) # velocidade linear distância / tempo
        # diff calcula a variação exemplo > X atual - X anterior
        # np.sqrt calcula a hipotenusa  raiz quadrada de x^2 - z^2
        #aceleração linear
        
        # loc trava na celula do df em que a linha é o mask, jogador
        # vel_posicao é a coluna que escreve o resultado
        # mask garante que estamos alterando apenas linhas referentes aquele jogador específico
        # divide distancia por tempo pra saber velocidade
        # detecta o speedhack brusco
        acel_lin = (vel_lin.diff() / dt).fillna(0.0)
        
        # Velocidade de Rotação (Aimbot)
        angulos = np.arctan2(player_df['dir_x'], player_df['dir_z'])
        # np.arctan2 transforma os vetores em angulos
        diff_ang = angulos.diff() # subtrai o angulo atual do anterior

        # Ajuste de rotação circular (evita erro do 359° -> 1°)
        diff_ang = np.abs(np.arctan2(np.sin(diff_ang), np.cos(diff_ang)))
        vel_rot = (diff_ang / dt).fillna(0.0)
        # divide angulo pelo tempo pra saber velocidade angular / rotação
        df.loc[mask, 'vel_rotacao'] = vel_rot

        # JITTER variação de velocidade angular (Variação da mira - Biometria Comportamental
        jitter = vel_rot.diff().abs().fillna(0.0)
        # inserindo de volta no data frame principal
        df.loc[mask, 'vel_posicao'] = vel_lin
        df.loc[mask, 'acel_linear'] = acel_lin
        df.loc[mask, 'jitter_mira'] = jitter
        df.loc[mask, 'vel_rotacao'] = vel_rot


    # SALVAMENTO
    cols_ia = ['pos_x', 'vel_posicao', 'acel_linear', 'vel_rotacao', 'jitter_mira', 'mirando']
    # removendo possíveis erros matemáticos ou valores infinitos
    df_final = df[cols_ia].replace([np.inf, -np.inf], np.nan).dropna()
    df_final.to_csv(output_path, index=False)
    print(f"Dataset salvo em: {output_path}")

preprocess_log(FILE_IN, FILE_OUT, INTERVALO_TEMPO)