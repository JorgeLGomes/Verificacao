# Verificacao — aplicacao unificada de verificacao do modelo Eta

Consolida, dirigida por **um unico arquivo de parametros**, a verificacao de:

- **Precipitacao** acumulada em 24 h → referencia **MERGE**
  (metricas continuas + categoricas ETS/BIAS + FSS, por limiar);
- **Campos continuos instantaneos** — Temperatura a 2 m, Pressao ao nivel do
  mar, vento a 10 m (U/V) e, preparado, geopotencial em 500 hPa → referencia
  **ERA5** (metricas continuas BIAS/MAE/RMSE/SCORR).

Todas as metricas sao calculadas sobre a **intersecao** de pontos validos,
estratificadas por regiao (Todo dominio / Continente / Oceano + Amazonia,
Nordeste, Sudeste, Sul) e por prazo de previsao. Modelos comparados: **jaci**
(linha continua) e **XC50** (tracejada).

## Fluxo em dois estagios

```
                config_verificacao.yaml
                          |
   (1) calculo   python verifica.py --config config_verificacao.yaml
                          |
                 verificacao.pkl  +  <componente>_*.csv     <- dados das figuras
                          |
   (2) figuras   python gera_figuras.py --binario .../verificacao.pkl
                          |
                 fig_*.png, mapas_*.png
```

O estagio (1) le os dados raw (NetCDF ou binctl), calcula tudo e grava um
**binario** e **CSVs**. O estagio (2) regenera as figuras **so a partir do
binario**, sem reprocessar os dados raw — ajustes de plot sao instantaneos.

## Arquivos

| Arquivo | Papel |
|---|---|
| `config_verificacao.yaml` | **parametros**: componentes, variaveis, referencias, regioes, prazos |
| `verifica.py` | driver de **calculo** → `verificacao.pkl` + CSVs |
| `gera_figuras.py` | **replot** a partir do binario |
| `nucleo.py` | metricas, regrid, mascaras, IO NetCDF, conversao de unidades, plots |
| `precip_engine.py` | motor de precipitacao (acumulo 24 h, janelas, ETS/FSS) |
| `leitor_binctl.py` | leitor nativo `.ctl/.bin` (GrADS) para `reader: binctl` |

## Uso

```bash
pip install -r requirements.txt          # global-land-mask e opcional (terra/mar)

# calculo (todos os componentes, ou um subconjunto):
python verifica.py --config config_verificacao.yaml
python verifica.py --config config_verificacao.yaml --componentes precipitacao t2m
python verifica.py --config config_verificacao.yaml --max-rodadas 2   # teste rapido
python verifica.py --config config_verificacao.yaml --jobs 8          # 8 processos
python verifica.py --config config_verificacao.yaml --jobs 0          # todos os cores

# figuras (todas, ou um subconjunto):
python gera_figuras.py --binario resultados_uni/verificacao.pkl
python gera_figuras.py --binario resultados_uni/verificacao.pkl --componentes t2m
```

## config_verificacao.yaml (resumo)

- `base` — raiz com `nc/`, `nc_oper/`, `nc_merge/`.
- `modelos` — `jaci` e `xc50`, com `subdir` e `reader` (`netcdf`/`binctl`).
- `referencias.merge` (NetCDF) e `referencias.era5` (arquivo unico multi-variavel).
- `componentes.*` — cada variavel a verificar declara:
  - `tipo`: `acum24` (precip) ou `instantaneo` (campos ERA5);
  - `referencia`: `merge` ou `era5`;
  - `unidade` final e, por modelo, `padrao`/`var`/`unidade` (+ `fator_para_mm`,
    `acumulado` na precip; `nivel` para campos em niveis, ex.: Z500);
  - `metricas`: `continuas` (sempre) + `categoricas`/`fss` (so precip) + `limiares`.
- `leads_horas` — prazos (horas) dos campos instantaneos; `janelas_precip` — 12/00Z.

### Paralelismo (`--jobs N`)

`verifica.py` divide as rodadas entre N processos (`--jobs N`; `0` = todos os
cores). Cada processo abre a sua propria referencia e acumula um subconjunto; os
acumuladores parciais sao mesclados no fim por estatisticas suficientes. O
resultado e' **identico** ao sequencial (validado: processar tudo de uma vez ==
processar em partes e mesclar). Como o FSS domina o custo, o ganho e' quase
linear com o numero de cores. No PBS, case `--jobs $NCPUS`.

### Mapas espaciais (media diaria por prazo, por mes)

A secao `mapas` da config controla os mapas espaciais: para cada dia de previsao
(prazo) e' gerada `mapa_<comp>_<modelo>_D<lead>.png` com **colunas**
[Todo periodo, Jan, Fev, Mar] e **linhas** [<referencia> media, Prev media,
Vies medio] — a linha da referencia e' o MERGE (precip) ou o ERA5 (demais).
Aplica-se a todos os campos: precipitacao, T2m, PNMM, U10, V10 e a magnitude
do vento a 10 m (`wspd10`, derivada de u e v).

Opcoes: `ativar`, `leads` (prazos a mapear; `null` = todos), `por_mes`, `meses`
(so meses completos, default `[1, 2, 3]` — descarta abril parcial) e `bordas`
(fronteiras de paises + estados do Brasil via cartopy). Em grades grandes,
restrinja `leads` para conter memoria/tamanho do binario (mapas em float32).

> **Fronteiras no cluster:** o cartopy baixa os shapefiles do Natural Earth na
> 1a vez e os guarda em cache (`~/.local/share/cartopy`). Se os nos de calculo
> nao tiverem internet, gere as figuras num no de login com acesso, ou
> pre-baixe os shapefiles; sem eles, as fronteiras sao simplesmente omitidas
> (as figuras saem sem travar).

O geopotencial em **500 hPa** ja esta no arquivo (componente `z500`
**comentado**): descomente e aponte o ERA5 em niveis de pressao + o campo 3D
do modelo (ou um `.ctl/.bin` ja no nivel 500) para ativa-lo.

## Leitor por variavel: NetCDF ou binctl

Cada modelo/variavel escolhe o `reader`:

- `netcdf` — le o `.nc` da rodada (auto-detecta variavel/coordenadas/tempo;
  seleciona nivel por `nivel`). Caminho testado.
- `binctl` — le o formato GrADS `.ctl/.bin` nativo (sem GrADS instalado), via
  `leitor_binctl.py`. So variaveis de superficie; para niveis, use um `.ctl/.bin`
  ja no nivel desejado. Reaproveita a logica do projeto ERA5 original.

## Metodologia (resumo)

- Regrid do modelo para a grade da **referencia** (MERGE 10 km / ERA5 0,25°),
  bilinear por padrao — evita criar informacao espuria.
- Precip: acumulo de 24 h (jaci ja acumulado ×1000 m→mm; XC50 somado dos passos),
  janelas 12Z-12Z e 00Z-00Z; `ETS`, `BIAS` categorico e `FSS` por limiar.
- Continuos: casamento pelo tempo valido (instantaneos 00/12Z); `SCORR` e a
  correlacao espacial media (Pearson por caso, media sobre os casos).
- Agregacao por estatisticas suficientes (soma correta de scores).

## Origem

Unifica os projetos `Verif_prec` (precipitacao × MERGE) e
`Script de verificacao` (campos continuos × ERA5).
