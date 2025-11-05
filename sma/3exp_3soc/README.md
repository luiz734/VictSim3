# 1 exp 1 soc: Explorador aleatório e um socorrista em DFS
Este exemplo apresenta um agente explorador que caminha aleatoriamente pelo ambiente. Ele constrói um mapa da região explorada contendo os obstáculos e as vítimas. Em seguida, o explorador passa o mapa para o socorrista. Posteriormente, o socorrista caminha usando a busca em profundidade (DFS) dentro da região descoberta, tentando resgatar as vítimas encontradas.

## Como usar:
- copie os arquivos explorer.py, rescuer.py e main.py para alguma 'pasta'
- copie a pasta config_ag_1 para a 'pasta'
- copie a pasta 'vs' para dentro da 'pasta'

Você deverá obter esta estrutura:
### pasta sma
- main.py
- rescuer.py
- explorer.py
- config_ag
  - explorer_1.txt    # configuração para o explorador 1
  - rescuer_1.txt     # configuração para o socorrista 1
- vs
  - abstract_agent.py
  - constants.py
  - environment.py
  - physical_agent.py
### datasets
#### env (arquivos que definem o ambiente)
- 12x12_10v (grid 12 por 12 com 10 vítimas)
  - env_config.txt (tamanho da janela, do grid, ...)
  - env_obst.txt (posições dos obstáculos)
  - env_victims.txt (posições das vítimas)
#### vic (datasets com os sinais vitais das vítimas)
- 10v
  - data.csv
