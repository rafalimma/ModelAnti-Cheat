import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURAÇÃO
# ==========================================
FILE_IN = r'log_profiles/script_2026-06-07_15-26-02-hacker-entropia.txt'
FILE_OUT = 'datasets/dataset_hacker_entropia.csv'
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

        # ------ JITTER --------
        # JITTER variação de velocidade angular (Variação da mira - Biometria Comportamental
        #A Anomalia: Se a vel_posicao é alta e o jitter_mira é quase nulo por mais de 3 segundos, significa que o jogador está travado em um objetivo fixo (lock-on).
        jitter = vel_rot.diff().abs().fillna(0.0)


        # ------ ENTROPIA DE MOVIMENTO --------
        #calcular a entropia de movimento
        # desvio padrão da direção, se o desvio padrão for próximo a zero enquanto a velocidade
        # é alta, a entropia é baixa
        '''
        A detecção não se baseia na localização do item, mas na anomalia de intenção. 
        Um jogador que mantém 99% de eficiência de trajeto (linha reta) por longos períodos, 
        combinada com baixa entropia de rotação (olhar fixo), apresenta um padrão estatístico incompatível com a exploração humana natural, 
        que é inerentemente caótica e ineficiente.
        '''
        entropia = vel_rot.rolling(window=5).std().fillna(0.0)

        # ------ EFICIÊNCIA DE TRAJETÓRIA --------
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

        '''
        Para saber se o Jogador A está olhando fixamente para o Jogador B através de um obstáculo, 
        usamos a Similaridade de Cosseno entre o vetor de visão do Jogador A e a direção real onde o Jogador B está.
        '''
        #O segredo aqui é que, após calcular as velocidades individuais de cada jogador, 
        #fazemos um segundo loop para comparar os jogadores entre si no mesmo tique de tempo.

    df['travado_em_player'] = 0.0
    df['perseguindo_player'] = 0
    for tempo_tique in df['tempo'].unique():
        tique_df = df[df['tempo'] == tempo_tique]
        # se ouver menos de 2 jogadores no tique não ha o que comparar
        if len(tique_df) < 2:
            continue

        for idx, player_atual in tique_df.iterrows():
            pid_atual = player_atual['player_id']
            px, pz = player_atual['pos_x'], player_atual['pos_z']
            dx, dz = player_atual['dir_x'], player_atual['dir_z']
            menor_desvio_angulo = 180.0
            esta_seguindo = 0

            # compara com todos os outros jogadores no mesmo tique
            for idx_inimigo, player_inimigo in tique_df.iterrows():
                if player_inimigo['player_id'] == pid_atual:
                    continue

                ix, iz = player_inimigo['pos_x'], player_inimigo['pos_z']

                #vetor do player atual até o inimigo
                vetor_inimigo_x = ix -px
                vetor_inimigo_z = iz -  pz
                dist_ate_inimigo = np.sqrt(vetor_inimigo_x**2 + vetor_inimigo_z**2)
                #ignora se estiver muito longe
                if dist_ate_inimigo > 300 or dist_ate_inimigo == 0:
                    continue

                # Normaliza o vetor do inimigo
                v_inimigo_x = vetor_inimigo_x / dist_ate_inimigo
                v_inimigo_z = vetor_inimigo_z / dist_ate_inimigo
                #calcula o angulo entre olhar do player (dx, dz) e a posição do inimigo
                # Similaridade de cosseno
                dot_product = (dx * v_inimigo_x) + (dz * v_inimigo_z)
                # Garante que o valor fique entre -1 e 1 para evitar erros matemáticos
                dot_product = np.clip(dot_product, -1.0, 1.0)

                angulo_desvio = np.degrees(np.arccos(dot_product))

                if angulo_desvio < menor_desvio_angulo:
                    menor_desvio_angulo = angulo_desvio
                
                # --- VERIFICAÇÃO DE PERSEGUIÇÃO/INTERCEPTAÇÃO ---
                # Se o player atual está se movendo rápido e a direção do movimento dele
                # aponta diretamente para a posição futura ou atual do inimigo
                if player_atual['vel_posicao'] > 4.0 and angulo_desvio < 10.0:
                    if player_atual['eficiencia_trajeto'] > 0.92:
                        esta_seguindo = 1
            # Atualiza o DataFrame principal com os resultados do cruzamento
            df.at[idx, 'travado_em_player'] = menor_desvio_angulo
            df.at[idx, 'perseguindo_player'] = esta_seguindo
            
    # Agora criamos rótulos baseados nas regras clássicas para testar seu trainer
    df['alerta_speedhack'] = (df['vel_posicao'] > 9.0).astype(int) # Acima da velocidade máxima de corrida humana
    df['alerta_aimbot'] = (df['vel_rotacao'] > 15.0).astype(int)   # Rotação sobre-humana instantânea
    df['alerta_lockon'] = ((df['jitter_mira'] < 0.001) & (df['vel_posicao'] > 0.0)).astype(int)
    df['alerta_esp_player'] = ((df['travado_em_player'] < 5.0) & (df['mirando'] == 1)).astype(int)



    # SALVAMENTO
    cols_ia = [
        'vel_posicao', 'acel_linear', 'vel_rotacao', 'jitter_mira', 
        'entropia_mov', 'eficiencia_trajeto', 'travado_em_player', 
        'perseguindo_player', 'mirando',
        'alerta_speedhack', 'alerta_aimbot', 'alerta_lockon', 'alerta_esp_player'
    ]
    # removendo possíveis erros matemáticos ou valores infinitos
    df_final = df[cols_ia].replace([np.inf, -np.inf], np.nan).dropna()
    df_final.to_csv(output_path, index=False)
    print(f"Dataset salvo em: {output_path}")

preprocess_log(FILE_IN, FILE_OUT, INTERVALO_TEMPO)