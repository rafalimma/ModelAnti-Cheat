import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURAÇÃO
# ==========================================
FILE_IN = r'D:\DayZServerProfiles\script_2026-hacker1.txt'
FILE_OUT = 'datasets/dataset_hacker.csv'
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
                # Removemos o prefixo "SCRIPT : " limpando tudo antes do DATA_LOG
                content = line.split("DATA_LOG")[-1] 
                
                # Agora dividimos pelo separador '|'
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
    df['entropia_mov'] = 0.0
    df['eficiencia_trajeto'] = 0.0

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

        # JITTER variação de velocidade angular (Variação da mira - Biometria Comportamental
        #A Anomalia: Se a vel_posicao é alta e o jitter_mira é quase nulo por mais de 3 segundos, significa que o jogador está travado em um objetivo fixo (lock-on).
        jitter = vel_rot.diff().abs().fillna(0.0)

        #calcular a entropia de movimento
        # desvio padrão da direção, se o desvio padrão for próximo a zero enquanto a velocidade
        # é alta, a entropia é baixa
        entropia = vel_rot.rolling(window=5).std().fillna(0.0)
        # eficiencia da trajetória (distancia reta x distancia percorrida)
        janela =20
        dist_acumulada = dist.rolling(window=janela).sum()
        # Distância em linha reta entre o ponto atual e o de 10 tiques atrás
        dx_10 = player_df['pos_x'] - player_df['pos_x'].shift(janela)
        dz_10 = player_df['pos_z'] - player_df['pos_z'].shift(janela)
        dist_reta_10 = np.sqrt(dx_10**2 + dz_10**2)
        eficiencia = (dist_reta_10 / dist_acumulada).fillna(0.0)
        df['entropia_mov'] = entropia
        # inserindo de volta no data frame principal
        df.loc[mask, 'vel_rotacao'] = vel_rot
        df.loc[mask, 'vel_posicao'] = vel_lin
        df.loc[mask, 'acel_linear'] = acel_lin
        df.loc[mask, 'jitter_mira'] = jitter
        df.loc[mask, 'vel_rotacao'] = vel_rot
        df.loc[mask, 'eficiencia_trajeto'] = eficiencia


    # SALVAMENTO
    cols_ia = ['pos_x', 'vel_posicao', 'acel_linear', 'vel_rotacao', 'jitter_mira','entropia_mov', 'eficiencia_trajeto', 'mirando']
    # removendo possíveis erros matemáticos ou valores infinitos
    df_final = df[cols_ia].replace([np.inf, -np.inf], np.nan).dropna()
    df_final.to_csv(output_path, index=False)
    print(f"Dataset salvo em: {output_path}")

preprocess_log(FILE_IN, FILE_OUT, INTERVALO_TEMPO)