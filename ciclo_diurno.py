#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ciclo_diurno.py
===============

Ciclo diurno medio de uma variavel (padrao T2m), comparando jaci, XC50 e ERA5
no MESMO grafico, com um painel por prazo de previsao (D+1, D+2, ...).

Para cada rodada e cada tempo valido de previsao:
  - le o campo do modelo (todas as horas do .nc), tira a media espacial no
    dominio do modelo;
  - le o ERA5 no mesmo tempo valido, media no mesmo dominio;
  - acumula por (fonte, prazo, hora-do-dia UTC).
No fim, plota a media -> ciclo diurno (T2m x hora) por prazo.

Reaproveita a config unificada (config_verificacao.yaml): base, subdir de cada
modelo, arquivo/variaveis do ERA5, filtro de meses e o componente da variavel
(padrao/var/unidade). NAO usa o binario de scores (precisa dos dados sub-diarios).

Uso:
  python ciclo_diurno.py --config config_verificacao.yaml --var t2m \
      --leads 1 3 5 7 --saida resultados_uni
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

import nucleo as N
import verifica as V   # reaproveita RefEra5, _resolve_arq, _lista_rodadas, _periodo_ok

COR_FONTE = {"jaci": "#1f77b4", "xc50": "#d62728", "ERA5": "black"}
LS_FONTE = {"jaci": "-", "xc50": "--", "ERA5": ":"}
MK_FONTE = {"jaci": "o", "xc50": "s", "ERA5": "^"}


def _bbox_grade(lats, lons):
    return [float(np.min(lons)), float(np.max(lons)),
            float(np.min(lats)), float(np.max(lats))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ciclo diurno medio (jaci/XC50/ERA5) por prazo.")
    ap.add_argument("--config", default="config_verificacao.yaml")
    ap.add_argument("--var", default="t2m", help="componente da config (t2m, pnmm, ...)")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 7],
                    help="prazos de previsao (dias) a plotar, um painel por prazo")
    ap.add_argument("--max-rodadas", type=int, default=0)
    ap.add_argument("--saida", default=None, help="pasta de saida (padrao: cfg.saida)")
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    comp = cfg["componentes"][args.var]
    unidade = comp.get("unidade", "")
    e5 = comp.get("era5", {})
    leads_alvo = set(args.leads)
    saida = args.saida or cfg.get("saida", "resultados_uni")
    os.makedirs(saida, exist_ok=True)

    ref = V.RefEra5(cfg["referencias"]["era5"]["arquivo"])
    LONe, LATe = np.meshgrid(ref.lons, ref.lats)
    bbox = None
    mask_era5 = None

    # acumuladores: soma e contagem por (fonte, lead, hora)
    soma = {}; cont = {}

    def add(fonte, lead, hora, val):
        if not np.isfinite(val):
            return
        k = (fonte, lead, int(hora))
        soma[k] = soma.get(k, 0.0) + val
        cont[k] = cont.get(k, 0) + 1

    for modelo, mspec in comp["modelos"].items():
        subdir = mspec.get("subdir", cfg["modelos"][modelo]["subdir"])
        runs = V._lista_rodadas(cfg["base"], subdir, cfg.get("init_glob", "*"),
                                args.max_rodadas)
        print(f"[{args.var}] {modelo}: {len(runs)} rodadas")
        for ri, run in enumerate(runs, 1):
            arq = V._resolve_arq(cfg, mspec, modelo, run, "netcdf", mspec.get("padrao"))
            if arq is None:
                continue
            init = datetime.strptime(run, "%Y%m%d%H")
            try:
                ds = N.abre(arq)
                var, latn, lonn, tn = N.nomes(ds, mspec.get("var"),
                                              prefer="tp2m|t2m|pslm|u10|v10")
                lats, lons = N.grade(ds, latn, lonn)
                da = ds[var]
                tempos = pd.to_datetime(ds[tn].values) if tn and tn in ds else None
            except Exception as ex:
                print(f"  [{run}] erro: {ex}"); continue
            if tempos is None:
                ds.close(); continue
            if bbox is None:                       # dominio do modelo (1a vez)
                bbox = _bbox_grade(lats, lons)
                mask_era5 = ((LATe >= bbox[2]) & (LATe <= bbox[3]) &
                             (LONe >= bbox[0]) & (LONe <= bbox[1]))
            umod = mspec.get("unidade")
            for it, valido in enumerate(tempos):
                fh = (valido - pd.Timestamp(init)) / timedelta(hours=1)
                if fh <= 0:
                    continue
                lead = int(np.ceil(fh / 24.0))
                if lead not in leads_alvo:
                    continue
                if not V._periodo_ok(cfg, valido):
                    continue
                arr = N.campo_2d(da, latn, lonn, tn, it)
                if umod:
                    arr = N.converte_unidade(arr, umod, unidade)
                add(modelo, lead, valido.hour, float(np.nanmean(arr)))
                # ERA5 no mesmo tempo valido, mesmo dominio
                co = ref.campo(valido, var=e5.get("var"),
                               unidade=e5.get("unidade"), para=unidade)
                if co is not None:
                    em = float(np.nanmean(np.where(mask_era5, co.dados, np.nan)))
                    add("ERA5", lead, valido.hour, em)
            ds.close()
            if ri % 10 == 0 or ri == len(runs):
                print(f"  {ri}/{len(runs)}")
    ref.close()

    # tabela media por (fonte, lead, hora)
    linhas = [dict(fonte=k[0], lead=k[1], hora=k[2], valor=soma[k] / cont[k],
                   n=cont[k]) for k in soma]
    df = pd.DataFrame(linhas)
    if df.empty:
        print("Nenhum dado casado. Confira config/var/leads."); return 1
    df.to_csv(os.path.join(saida, f"ciclo_diurno_{args.var}.csv"), index=False)

    # figura: um painel por prazo; linhas jaci/XC50/ERA5
    leads = sorted(df.lead.unique())
    ncol = min(3, len(leads)); nrow = int(np.ceil(len(leads) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.4 * nrow),
                            squeeze=False)
    fontes = [f for f in ["jaci", "xc50", "ERA5"] if f in df.fonte.unique()]
    for k, lead in enumerate(leads):
        ax = axs.ravel()[k]
        for fonte in fontes:
            sub = df[(df.lead == lead) & (df.fonte == fonte)].sort_values("hora")
            if sub.empty:
                continue
            ax.plot(sub.hora, sub.valor, LS_FONTE[fonte], marker=MK_FONTE[fonte],
                    color=COR_FONTE[fonte], lw=1.8, markersize=4, label=fonte)
        ax.set_title(f"D+{lead}"); ax.set_xlabel("Hora (UTC)")
        ax.set_ylabel(f"{args.var} ({unidade})"); ax.grid(alpha=0.3)
        ax.set_xticks(sorted(df.hora.unique()))
    for k in range(len(leads), nrow * ncol):
        axs.ravel()[k].axis("off")
    h = [plt.Line2D([0], [0], color=COR_FONTE[f], linestyle=LS_FONTE[f],
                    marker=MK_FONTE[f], label=f) for f in fontes]
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.99),
               ncol=len(h), fontsize=10, framealpha=0.9)
    fig.suptitle(f"Ciclo diurno medio - {args.var} (media no dominio do Eta) "
                 f"por prazo", fontsize=13, x=0.5, y=0.995)
    cam = os.path.join(saida, f"ciclo_diurno_{args.var}.png")
    fig.savefig(cam, dpi=130); plt.close(fig)
    print(f"\nFigura: {cam}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
