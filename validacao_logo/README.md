# Validação Leave-One-Group-Out

Este teste mantém uma partida inteira fora do treinamento em cada rodada. O
`trainer_arvore_decisao.py` original não é alterado.

No PowerShell:

```powershell
cd D:\ModelAnti-Cheat
.\venv\Scripts\Activate.ps1
python .\validacao_logo\validacao_logo_arvore.py `
  --manifesto .\partidas_rotuladas\rotulos_treino_geral.csv `
  --agrupar-por partida_id `
  --saida .\models\arvore_logo_4_partidas
```

O script gera métricas por partida, previsões fora da amostra, matriz de
confusão acumulada, importâncias por rodada e um modelo final treinado com todas
as partidas. As colunas de arma são mantidas apenas para auditoria e não são
usadas como features.
