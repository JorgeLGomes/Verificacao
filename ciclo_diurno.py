#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ciclo_diurno.py — ciclo diurno medio de uma variavel, comparando jaci, XC50 e
ERA5, separado por horizonte de previsao (D+n) E por regiao.

Regioes (mesmas do resto do app, via nucleo.constroi_mascaras):
  Todo, Continente, Oceano (land mask) + regioes de um shapefile
  (regioes.shapefile no config; p.ex. as 12 Regioes Hidrograficas do Brasil).
  Cada figura de regiao inclui um mini-mapa com a regiao em destaque.

Fluxo:
  - calcula(): le os NetCDF do(s) modelo(s) + ERA5, acumula somas por
    (fonte, regiao, prazo, hora) e grava um CSV (ciclo_diurno_<var>.csv) para
    nao ter de reprocessar os dados raw ao replotar.
  - plota(): uma figura por regiao; em cada figura, um painel por prazo e
    linhas jaci (azul, -), XC50 (vermelho, --), ERA5 (preto, :).

Otimizacoes:
  - media espacial por arquivo/regiao VETORIZADA em numpy (uma leitura por
    arquivo; todas as regioes de uma vez);
  - mascaras de regiao construidas UMA vez por grade (cache por assinatura),
    evitando refazer o land mask a cada arquivo;
  - serie de dominio do ERA5 por regiao lida UMA vez por processo;
  - paralelizacao opcional por rodadas com --jobs N (cada processo acumula um
    pedaco e as somas sao mescladas no fim). Cada processo abre a SUA propria
    referencia ERA5, entao use um --jobs modesto para nao saturar o Lustre.

Uso:
  python ciclo_diurno.py --var t2m --leads 1 3 5 7 --jobs 4
  python ciclo_diurno.py --var t2m --regioes Todo Sudeste Sul --replot
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
import verifica as V

COR_FONTE = {"jaci": "#1f77b4", "xc50": "#d62728", "ERA5": "black"}
LS_FONTE = {"jaci": "-", "xc50": "--", "ERA5": ":"}
MK_FONTE = {"jaci": "o", "xc50": "s", "ERA5": "^"}
ORDEM_REG = ["Todo", "Continente", "Oceano"]   # demais regioes seguem o shapefile
REG_DOMINIO = {                                # regioes baseadas no dominio do Eta
    "Todo": "todo o dominio do Eta",
    "Continente": "continente (dominio do Eta)",
    "Oceano": "oceano (dominio do Eta)",
}


def _rotulo_regiao(regiao):
    """Texto da regiao para o titulo do grafico."""
    if regiao in REG_DOMINIO:
        return REG_DOMINIO[regiao]
    return f"macro bacia hidrografica: {regiao}"


def _carrega_geoms(rcfg):
    """Retorna (tipo, dict). tipo='shp' -> {nome: geometria shapely};
    tipo='box' -> {nome: (la0,la1,lo0,lo1)}; ou (None, {})."""
    shp = rcfg.get("shapefile")
    if shp and os.path.isfile(shp):
        try:
            regs = N._le_shapefile(shp, rcfg.get("campo_nome"))
            # simplifica p/ o mini-mapa (poligonos das RHs sao muito detalhados)
            return "shp", {nome: geom.simplify(0.05, preserve_topology=False)
                           for nome, geom in regs}
        except Exception:
            pass
    caixas = rcfg.get("caixas_br") or {}
    if caixas:
        return "box", {k: tuple(v) for k, v in caixas.items()}
    return None, {}


def _desenha_geom(ax, geom, **kw):
    gs = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for g in gs:
        x, y = g.exterior.xy
        ax.fill(x, y, **kw)


def _mapa_dominio(ax, dom, modo):
    """Mini-mapa do dominio do Eta: pinta continente/oceano ou (Todo) nada.
    dom = (lats, lons, land[bool] | None)."""
    if not dom:
        ax.axis("off"); return
    from matplotlib.colors import ListedColormap
    lats, lons, land = dom
    hi = np.zeros((len(lats), len(lons)), dtype=float)   # 0 = dominio (cinza)
    if land is not None and getattr(land, "size", 0):
        if modo == "Continente":
            hi = land.astype(float)                      # 1 = continente (vermelho)
        elif modo == "Oceano":
            hi = (~land).astype(float)                   # 1 = oceano (vermelho)
    ax.pcolormesh(lons, lats, hi, cmap=ListedColormap(["#e6e6e6", "#d62728"]),
                  vmin=0, vmax=1, shading="auto")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(REG_DOMINIO.get(modo, modo), fontsize=9)


def _mapa_regiao(ax, tipo, geoms, regiao, dom=None):
    """Mini-mapa: dominio do Eta (Todo/Continente/Oceano) ou bacia em destaque."""
    if regiao in REG_DOMINIO:
        _mapa_dominio(ax, dom, regiao); return
    if not geoms:
        ax.axis("off"); return
    xs, ys = [], []
    for nome, g in geoms.items():
        if tipo == "shp":
            b = g.bounds; xs += [b[0], b[2]]; ys += [b[1], b[3]]
            _desenha_geom(ax, g, facecolor="#e6e6e6", edgecolor="#888",
                          linewidth=0.4, zorder=1)
        else:
            la0, la1, lo0, lo1 = g; xs += [lo0, lo1]; ys += [la0, la1]
            ax.fill([lo0, lo1, lo1, lo0], [la0, la0, la1, la1],
                    facecolor="#e6e6e6", edgecolor="#888", linewidth=0.4, zorder=1)
    dest = geoms.get(regiao)
    if dest is not None:
        if tipo == "shp":
            _desenha_geom(ax, dest, facecolor="#d62728", edgecolor="#7a0000",
                          linewidth=0.6, alpha=0.85, zorder=3)
        else:
            la0, la1, lo0, lo1 = dest
            ax.fill([lo0, lo1, lo1, lo0], [la0, la0, la1, la1],
                    facecolor="#d62728", edgecolor="#7a0000", alpha=0.85, zorder=3)
    if xs and ys:
        mx = (max(xs) - min(xs)) * 0.03 + 0.5; my = (max(ys) - min(ys)) * 0.03 + 0.5
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"macro bacia: {regiao}", fontsize=9)


def _assinatura_grade(lats, lons):
    return (len(lats), len(lons), float(lats[0]), float(lats[-1]),
            float(lons[0]), float(lons[-1]))


def _get_masks(lats, lons, rcfg, sel_norm, cache):
    """Mascaras 2D (bool) por regiao, cacheadas por assinatura de grade.
    Usa poligonos do shapefile (rcfg['shapefile']) se definido, senao caixas."""
    sig = _assinatura_grade(lats, lons)
    if sig not in cache:
        masks, _ = N.constroi_mascaras(
            np.asarray(lats), np.asarray(lons), rcfg.get("caixas_br"),
            shapefile=rcfg.get("shapefile"), campo_nome=rcfg.get("campo_nome"))
        if sel_norm:
            masks = {k: v for k, v in masks.items()
                     if N.slug_regiao(k).upper() in sel_norm}
        cache[sig] = masks
    return cache[sig]


def _medias_regioes(arr, masks):
    """arr (T,ny,nx) -> dict regiao -> array(T) com a media espacial (ignora NaN)."""
    valid = np.isfinite(arr)
    out = {}
    for nome, m in masks.items():
        m3 = m[None, :, :]
        sel = valid & m3
        cnt = sel.sum(axis=(1, 2))
        s = np.where(sel, arr, 0.0).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[nome] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out


def _campo_np(ds, v, latn, lonn, tn):
    """DataArray -> numpy (T,ny,nx) alinhado com (lats, lons)."""
    da = ds[v]
    sel = {d: 0 for d in da.dims if d not in (latn, lonn, tn)}
    da = da.isel(sel) if sel else da
    da = da.transpose(tn, latn, lonn)
    return da.values.astype(float)


def _prep_era5(ref, e5, unidade, rcfg, sel_norm, cache):
    """Series de dominio do ERA5 por regiao (uma leitura, todas as regioes)."""
    var = e5.get("var")
    masks = _get_masks(ref.lats, ref.lons, rcfg, sel_norm, cache)
    da = ref.ds[var]
    sel = {d: 0 for d in da.dims if d not in (ref.latn, ref.lonn) and d != getattr(ref, "tn", None)}
    # descobre o nome do tempo do ERA5
    tn = None
    for cand in ("time", "valid_time", "t"):
        if cand in da.dims:
            tn = cand; break
    if tn is None:
        tn = [d for d in da.dims if d not in (ref.latn, ref.lonn)]
        tn = tn[0] if tn else None
    sel = {d: 0 for d in da.dims if d not in (ref.latn, ref.lonn, tn)}
    da = (da.isel(sel) if sel else da).transpose(tn, ref.latn, ref.lonn)
    arr = da.values.astype(float)
    if e5.get("unidade") and unidade:
        arr = N.converte_unidade(arr, e5.get("unidade"), unidade)
    med = _medias_regioes(arr, masks)
    te5 = np.asarray(ref.tempos, dtype="datetime64[ns]")
    return med, te5


def _acumula_chunk(cfg, var, unidade, e5, leads_alvo, sel_norm, tarefas):
    """Acumula soma/cont de (fonte,regiao,lead,hora) para uma lista de tarefas
    (modelo, mspec, run). Abre a SUA propria referencia ERA5 (uma vez)."""
    ref = V.RefEra5(cfg["referencias"]["era5"]["arquivo"])
    rcfg = cfg.get("regioes", {})     # caixas_br / shapefile / campo_nome
    cache = {}                 # assinatura de grade -> masks
    e5_med = None; te5 = None  # series ERA5 por regiao (uma vez)
    soma = {}; cont = {}

    def add(fonte, regiao, lead, hora, val):
        if not np.isfinite(val):
            return
        k = (fonte, regiao, lead, int(hora))
        soma[k] = soma.get(k, 0.0) + val
        cont[k] = cont.get(k, 0) + 1

    for modelo, mspec, run in tarefas:
        arq = V._resolve_arq(cfg, mspec, modelo, run, "netcdf", mspec.get("padrao"))
        if arq is None:
            continue
        umod = mspec.get("unidade")
        init = datetime.strptime(run, "%Y%m%d%H")
        try:
            ds = N.abre(arq)
            v, latn, lonn, tn = N.nomes(ds, mspec.get("var"),
                                        prefer="tp2m|t2m|pslm|u10|v10")
            lats, lons = N.grade(ds, latn, lonn)
            tempos = pd.to_datetime(ds[tn].values) if tn and tn in ds else None
            if tempos is None:
                ds.close(); continue
            arr = _campo_np(ds, v, latn, lonn, tn)
            if umod:
                arr = N.converte_unidade(arr, umod, unidade)
        except Exception as ex:
            print(f"  [{run}] erro: {ex}"); continue
        ds.close()
        masks = _get_masks(lats, lons, rcfg, sel_norm, cache)
        med_reg = _medias_regioes(arr, masks)
        if e5_med is None:
            e5_med, te5 = _prep_era5(ref, e5, unidade, rcfg, sel_norm, cache)
        for it, valido in enumerate(tempos):
            fh = (valido - pd.Timestamp(init)) / timedelta(hours=1)
            if fh <= 0:
                continue
            lead = int(np.ceil(fh / 24.0))
            if lead not in leads_alvo or not V._periodo_ok(cfg, valido):
                continue
            for regiao in masks:
                add(modelo, regiao, lead, valido.hour, med_reg[regiao][it])
            j = int(np.argmin(np.abs(te5 - np.datetime64(valido))))
            if abs((pd.Timestamp(te5[j]) - valido).total_seconds()) <= 3 * 3600:
                for regiao in masks:
                    add("ERA5", regiao, lead, valido.hour, e5_med[regiao][j])
    ref.close()
    return soma, cont


def _worker_cd(payload):
    return _acumula_chunk(*payload)


def _salva_dominio(cfg, comp, saida, var):
    """Salva grade + mascara de terra do dominio do Eta (1o arquivo de modelo)
    em .npz, para o mini-mapa de Todo/Continente/Oceano no replot."""
    caminho = os.path.join(saida, f"ciclo_diurno_{var}_dominio.npz")
    for modelo, mspec in comp["modelos"].items():
        subdir = mspec.get("subdir", cfg["modelos"][modelo]["subdir"])
        runs = V._lista_rodadas(cfg["base"], subdir, cfg.get("init_glob", "*"), 0)
        for run in runs:
            arq = V._resolve_arq(cfg, mspec, modelo, run, "netcdf",
                                 mspec.get("padrao"))
            if not arq:
                continue
            try:
                ds = N.abre(arq)
                v, latn, lonn, tn = N.nomes(ds, mspec.get("var"),
                                            prefer="tp2m|t2m|pslm|u10|v10")
                lats, lons = N.grade(ds, latn, lonn); ds.close()
                masks, _ = N.constroi_mascaras(lats, lons)   # so land (sem shp)
                land = masks.get("Continente")
                sla = max(1, len(lats) // 300); slo = max(1, len(lons) // 300)
                np.savez(caminho, lats=np.asarray(lats)[::sla],
                         lons=np.asarray(lons)[::slo],
                         land=(land[::sla, ::slo] if land is not None
                               else np.zeros((0, 0), bool)))
                return caminho
            except Exception:
                continue
    return None


def _carrega_dominio(saida, var):
    caminho = os.path.join(saida, f"ciclo_diurno_{var}_dominio.npz")
    if not os.path.isfile(caminho):
        return None
    d = np.load(caminho)
    land = d["land"]
    return (d["lats"], d["lons"], land if land.size else None)


def calcula(cfg, args, saida):
    comp = cfg["componentes"][args.var]
    unidade = comp.get("unidade", "")
    e5 = comp.get("era5", {})
    leads_alvo = set(args.leads)
    sel_norm = ({N.slug_regiao(r).upper() for r in args.regioes}
                if args.regioes else None)

    tarefas = []
    for modelo, mspec in comp["modelos"].items():
        subdir = mspec.get("subdir", cfg["modelos"][modelo]["subdir"])
        runs = V._lista_rodadas(cfg["base"], subdir, cfg.get("init_glob", "*"),
                                args.max_rodadas)
        print(f"[{args.var}] {modelo}: {len(runs)} rodadas")
        for run in runs:
            tarefas.append((modelo, mspec, run))

    soma = {}; cont = {}

    def _merge(sc):
        s, c = sc
        for k, v in s.items():
            soma[k] = soma.get(k, 0.0) + v
        for k, v in c.items():
            cont[k] = cont.get(k, 0) + v

    jobs = int(getattr(args, "jobs", 1) or 1)
    if jobs > 1 and len(tarefas) > 1:
        import multiprocessing as mp
        nj = min(jobs, len(tarefas))
        chunks = [tarefas[i::nj] for i in range(nj)]      # round-robin
        payloads = [(cfg, args.var, unidade, e5, leads_alvo, sel_norm, ch)
                    for ch in chunks]
        print(f"  paralelo: {nj} processos ({len(tarefas)} tarefas)")
        with mp.Pool(nj) as pool:
            for sc in pool.map(_worker_cd, payloads):
                _merge(sc)
    else:
        print(f"  sequencial ({len(tarefas)} tarefas)")
        _merge(_acumula_chunk(cfg, args.var, unidade, e5, leads_alvo,
                              sel_norm, tarefas))

    df = pd.DataFrame([dict(fonte=k[0], regiao=k[1], lead=k[2], hora=k[3],
                            valor=soma[k] / cont[k], n=cont[k]) for k in soma])
    csv = os.path.join(saida, f"ciclo_diurno_{args.var}.csv")
    if not df.empty:
        df.sort_values(["regiao", "fonte", "lead", "hora"]).to_csv(csv, index=False)
        print(f"CSV: {csv}")
    _salva_dominio(cfg, comp, saida, args.var)   # grade p/ mini-mapa do dominio
    return df, unidade


def plota(df, var, unidade, saida, rcfg=None):
    if df.empty:
        print("Sem dados para plotar."); return
    regioes = [r for r in ORDEM_REG if r in df.regiao.unique()]
    regioes += [r for r in df.regiao.unique() if r not in regioes]
    fontes_all = [f for f in ["jaci", "xc50", "ERA5"] if f in df.fonte.unique()]
    tipo, geoms = _carrega_geoms(rcfg or {})
    dom = _carrega_dominio(saida, var)
    for regiao in regioes:
        dr = df[df.regiao == regiao]
        if dr.empty:
            continue
        leads = sorted(dr.lead.unique())
        nslots = len(leads) + 1                    # +1 painel p/ o mini-mapa
        ncol = min(3, nslots); nrow = int(np.ceil(nslots / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.4 * nrow),
                                squeeze=False)
        for k, lead in enumerate(leads):
            ax = axs.ravel()[k]
            for fonte in fontes_all:
                sub = dr[(dr.lead == lead) & (dr.fonte == fonte)].sort_values("hora")
                if sub.empty:
                    continue
                ax.plot(sub.hora, sub.valor, LS_FONTE[fonte], marker=MK_FONTE[fonte],
                        color=COR_FONTE[fonte], lw=1.8, markersize=4, label=fonte)
            ax.set_title(f"D+{lead}"); ax.set_xlabel("Hora (UTC)")
            ax.set_ylabel(f"{var} ({unidade})"); ax.grid(alpha=0.3)
            ax.set_xticks(sorted(dr.hora.unique()))
        # mini-mapa da regiao no proximo slot; demais slots desligados
        ax_map = axs.ravel()[len(leads)]
        _mapa_regiao(ax_map, tipo, geoms, regiao, dom)
        for k in range(len(leads) + 1, nrow * ncol):
            axs.ravel()[k].axis("off")
        h = [plt.Line2D([0], [0], color=COR_FONTE[f], linestyle=LS_FONTE[f],
                        marker=MK_FONTE[f], label=f) for f in fontes_all]
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                   ncol=len(h), fontsize=10, framealpha=0.9)
        fig.suptitle(f"Ciclo diurno medio - {var} - {_rotulo_regiao(regiao)} - por prazo",
                     fontsize=13, x=0.5, y=0.995)
        cam = os.path.join(saida, f"ciclo_diurno_{var}_{N.slug_regiao(regiao)}.png")
        fig.savefig(cam, dpi=130); plt.close(fig)
        print(f"Figura: {cam}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Ciclo diurno medio (jaci/XC50/ERA5) por prazo e por regiao.")
    ap.add_argument("--config", default="config_verificacao.yaml")
    ap.add_argument("--var", default="t2m")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--regioes", nargs="+", default=None,
                    help="subconjunto de regioes (padrao: todas). Ex.: Todo Sudeste")
    ap.add_argument("--max-rodadas", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="processos paralelos (0 = min(cpu,8)); cada um abre "
                         "sua propria ERA5, use valores modestos.")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args(argv)
    if args.jobs == 0:
        import multiprocessing as mp
        args.jobs = min(mp.cpu_count(), 8)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    comps = cfg.get("componentes", {})
    if args.var not in comps:
        print(f"ERRO: componente '{args.var}' nao existe no config. "
              f"Disponiveis: {', '.join(comps.keys())}")
        print("  (dica: a pressao ao nivel do mar e o componente 'pnmm'; "
              "'pslm' e apenas o prefixo do arquivo pslm_*.nc)")
        return 2
    saida = args.saida or cfg.get("saida", "resultados_uni")
    os.makedirs(saida, exist_ok=True)
    unidade = comps[args.var].get("unidade", "")
    rcfg = cfg.get("regioes", {})
    csv = os.path.join(saida, f"ciclo_diurno_{args.var}.csv")
    if args.replot:
        if not os.path.isfile(csv):
            print(f"ERRO: CSV nao encontrado: {csv}."); return 1
        df = pd.read_csv(csv)
        if "regiao" not in df.columns:
            df["regiao"] = "Todo"          # compat com CSV antigo (so dominio)
        if args.leads:
            df = df[df.lead.isin(args.leads)]
        if args.regioes:
            sel = {N.slug_regiao(r).upper() for r in args.regioes}
            df = df[df.regiao.map(lambda r: N.slug_regiao(r).upper() in sel)]
        plota(df, args.var, unidade, saida, rcfg)
        return 0
    df, unidade = calcula(cfg, args, saida)
    plota(df, args.var, unidade, saida, rcfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
