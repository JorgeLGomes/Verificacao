#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nucleo.py — nucleo compartilhado da aplicacao de verificacao.

Reune o que e' comum a todos os componentes (precipitacao x MERGE e campos
continuos x ERA5): leitura NetCDF, conversao de unidades, regrid, mascaras de
regiao, acumuladores de metricas (continuas com SCORR, categoricas ETS/BIAS,
FSS) e funcoes de plotagem no estilo do relatorio.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

CAND_LAT = ["lat", "latitude", "y", "Y", "XLAT", "g0_lat_0", "rlat"]
CAND_LON = ["lon", "longitude", "x", "X", "XLONG", "g0_lon_1", "rlon"]
CAND_TIME = ["time", "valid_time", "t", "Time", "times", "tempo"]
CAND_LEV = ["level", "lev", "plev", "isobaricInhPa", "pressure", "levelist"]


# ==========================================================================
# CAMPO / REGRID / UNIDADES
# ==========================================================================
@dataclass
class Campo:
    dados: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    nome: str = ""

    def __post_init__(self):
        self.dados = np.asarray(self.dados, dtype=float)
        self.lats = np.asarray(self.lats, dtype=float)
        self.lons = np.asarray(self.lons, dtype=float)


def norm_lons(lons):
    lons = np.asarray(lons, dtype=float)
    return np.where(lons > 180.0, lons - 360.0, lons)


def converte_unidade(arr, de, para):
    """Converte K<->C e Pa<->hPa (e m/s inalterado)."""
    if de == para or not de or not para:
        return arr
    conv = {("K", "C"): lambda x: x - 273.15,
            ("C", "K"): lambda x: x + 273.15,
            ("Pa", "hPa"): lambda x: x / 100.0,
            ("hPa", "Pa"): lambda x: x * 100.0,
            ("m", "mm"): lambda x: x * 1000.0,
            ("mm", "m"): lambda x: x / 1000.0}
    f = conv.get((de, para))
    if f is None:
        raise ValueError(f"conversao nao suportada: {de} -> {para}")
    return f(arr)


def regrid(origem: Campo, destino: Campo, metodo="linear") -> Campo:
    from scipy.interpolate import RegularGridInterpolator
    lats_o, dados = origem.lats, origem.dados
    if lats_o[0] > lats_o[-1]:
        lats_o = lats_o[::-1]; dados = dados[::-1, :]
    lons_o = origem.lons
    if lons_o[0] > lons_o[-1]:
        lons_o = lons_o[::-1]; dados = dados[:, ::-1]
    it = RegularGridInterpolator((lats_o, lons_o), dados, method=metodo,
                                 bounds_error=False, fill_value=np.nan)
    LON, LAT = np.meshgrid(destino.lons, destino.lats)
    z = it(np.column_stack([LAT.ravel(), LON.ravel()])).reshape(LAT.shape)
    return Campo(z, destino.lats, destino.lons, nome=origem.nome + "_rg")


# ==========================================================================
# IO NetCDF
# ==========================================================================
def _acha(ds, cands):
    for c in cands:
        if c in ds.coords or c in ds.variables or c in getattr(ds, "sizes", {}):
            return c
    return None


def abre(caminho):
    import xarray as xr
    return xr.open_dataset(caminho, decode_times=True)


def nomes(ds, var=None, prefer=r"prec|pr|rain|acpc|tp"):
    latn = _acha(ds, CAND_LAT); lonn = _acha(ds, CAND_LON); tn = _acha(ds, CAND_TIME)
    if var is None:
        cand = [v for v in ds.data_vars if ds[v].ndim >= 2]
        pref = [v for v in cand if re.search(prefer, v, re.I)]
        var = (pref or cand)[0]
    return var, latn, lonn, tn


def grade(ds, latn, lonn):
    return ds[latn].values, norm_lons(ds[lonn].values)


def campo_2d(da, latn, lonn, tn, it=0, nivel=None):
    """Extrai campo 2D (lat,lon) no indice de tempo 'it' e nivel opcional."""
    sel = {}
    for d in da.dims:
        if d in (latn, lonn):
            continue
        if d == tn:
            sel[d] = it
        elif nivel is not None and d in CAND_LEV:
            # seleciona o nivel mais proximo
            vals = da[d].values
            sel[d] = int(np.argmin(np.abs(vals - nivel)))
        else:
            sel[d] = 0
    da2 = da.isel(sel) if sel else da
    return da2.transpose(latn, lonn).values


def passo_horas(tempos):
    if tempos is None or len(tempos) < 2:
        return None
    d = pd.Series(tempos).diff().dropna()
    return d.dt.total_seconds().mode().iloc[0] / 3600.0


# ==========================================================================
# MASCARAS DE REGIAO
# ==========================================================================
CAIXAS_BR = {
    "Amazonia": (-7.0, -1.0, -68.0, -58.0),
    "Nordeste": (-12.0, -5.0, -42.0, -35.0),
    "Sudeste":  (-24.0, -18.0, -48.0, -40.0),
    "Sul":      (-32.0, -26.0, -57.0, -49.0),
}


def constroi_mascaras(lats, lons, caixas=None):
    caixas = CAIXAS_BR if caixas is None else caixas
    LON, LAT = np.meshgrid(lons, lats)
    masks = {"Todo": np.ones(LAT.shape, dtype=bool)}
    tem_terra = False
    try:
        from global_land_mask import globe
        land = globe.is_land(LAT, LON)
        masks["Continente"] = land
        masks["Oceano"] = ~land
        tem_terra = True
    except Exception as e:
        print(f"  [aviso] global_land_mask indisponivel ({e}); "
              f"Continente/Oceano ignorados.")
    for nome, (la0, la1, lo0, lo1) in caixas.items():
        masks[nome] = (LAT >= la0) & (LAT <= la1) & (LON >= lo0) & (LON <= lo1)
    return masks, tem_terra


# ==========================================================================
# ACUMULADORES DE METRICAS
# ==========================================================================
class AccCont:
    """BIAS/MAE/RMSE (estat. suficientes) + SCORR (corr espacial media por caso)."""
    def __init__(self):
        self.n = 0.0; self.sdif = 0.0; self.sabs = 0.0; self.sdif2 = 0.0
        self.scorr = 0.0; self.ncasos = 0

    def add_caso(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        if m.sum() == 0:
            return
        p, o = p[m], o[m]; d = p - o
        self.n += p.size
        self.sdif += d.sum(); self.sabs += np.abs(d).sum(); self.sdif2 += (d * d).sum()
        if p.size > 2 and p.std() > 0 and o.std() > 0:
            self.scorr += np.corrcoef(p, o)[0, 1]; self.ncasos += 1

    def merge(self, o):
        self.n += o.n; self.sdif += o.sdif; self.sabs += o.sabs
        self.sdif2 += o.sdif2; self.scorr += o.scorr; self.ncasos += o.ncasos

    def scores(self):
        if self.n == 0:
            return dict(BIAS=np.nan, MAE=np.nan, RMSE=np.nan, SCORR=np.nan, n=0)
        return dict(BIAS=self.sdif / self.n, MAE=self.sabs / self.n,
                    RMSE=np.sqrt(self.sdif2 / self.n),
                    SCORR=(self.scorr / self.ncasos) if self.ncasos else np.nan,
                    n=int(self.n))


class AccCategoria:
    def __init__(self, limiares):
        self.limiares = list(limiares)
        self.abcd = {t: [0, 0, 0, 0] for t in limiares}

    def add(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        p, o = p[m], o[m]
        for t in self.limiares:
            pp, oo = p >= t, o >= t
            self.abcd[t][0] += int(np.sum(pp & oo))
            self.abcd[t][1] += int(np.sum(pp & ~oo))
            self.abcd[t][2] += int(np.sum(~pp & oo))
            self.abcd[t][3] += int(np.sum(~pp & ~oo))

    def merge(self, o):
        for t in self.limiares:
            for i in range(4):
                self.abcd[t][i] += o.abcd[t][i]

    def scores(self):
        out = []
        for t in self.limiares:
            a, b, c, d = self.abcd[t]; n = a + b + c + d
            sd = lambda x, y: x / y if y else np.nan
            a_ref = sd((a + b) * (a + c), n)
            esp = sd((a + b) * (a + c) + (c + d) * (b + d), n)
            out.append(dict(limiar_mm=t, hits=a, false_alarms=b, misses=c,
                            correct_neg=d, n=n, POD=sd(a, a + c), FAR=sd(b, a + b),
                            CSI=sd(a, a + b + c), FBIAS=sd(a + b, a + c),
                            ETS=sd(a - a_ref, a + b + c - a_ref),
                            HSS=sd(a + d - esp, n - esp), acuracia=sd(a + d, n)))
        return out


def _fracoes(campo_bin, n):
    if n <= 0:
        return np.asarray(campo_bin, dtype=float)
    from scipy.ndimage import uniform_filter
    return uniform_filter(np.asarray(campo_bin, float), size=2 * n + 1,
                          mode="constant", cval=0.0)


class AccFSS:
    def __init__(self, limiares, escalas):
        self.limiares = list(limiares); self.escalas = list(escalas)
        self.num = {}; self.den = {}; self.cnt = {}
        for t in limiares:
            for s in escalas:
                self.num[(t, s)] = 0.0; self.den[(t, s)] = 0.0; self.cnt[(t, s)] = 0.0

    def add(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        npix = int(m.sum())
        for t in self.limiares:
            pb = (np.where(m, p, 0.0) >= t) & m
            ob = (np.where(m, o, 0.0) >= t) & m
            # atalho: sem evento em nenhum dos campos -> fracoes todas 0
            # (contribuicao nula ao num/den); so acumula a contagem. Identico.
            if not pb.any() and not ob.any():
                for s in self.escalas:
                    self.cnt[(t, s)] += npix
                continue
            for s in self.escalas:
                nn = s // 2
                fp = _fracoes(pb.astype(float), nn)[m]
                fo = _fracoes(ob.astype(float), nn)[m]
                self.num[(t, s)] += np.sum((fp - fo) ** 2)
                self.den[(t, s)] += np.sum(fp ** 2) + np.sum(fo ** 2)
                self.cnt[(t, s)] += fp.size

    def merge(self, o):
        for k in self.num:
            self.num[k] += o.num[k]; self.den[k] += o.den[k]; self.cnt[k] += o.cnt[k]

    def scores(self):
        out = []
        for t in self.limiares:
            for s in self.escalas:
                den = self.den[(t, s)]
                out.append(dict(limiar_mm=t, escala_px=s,
                                FSS=(1 - self.num[(t, s)] / den) if den > 0 else np.nan,
                                n=int(self.cnt[(t, s)])))
        return out


class Mapas:
    def __init__(self, grade_campo: Campo):
        s = (grade_campo.lats.size, grade_campo.lons.size)
        self.grade = grade_campo
        self.sp = np.zeros(s); self.so = np.zeros(s)
        self.sdif = np.zeros(s); self.n = np.zeros(s)

    def add(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        self.sp[m] += p[m]; self.so[m] += o[m]
        self.sdif[m] += (p[m] - o[m]); self.n[m] += 1

    def merge(self, o):
        self.sp += o.sp; self.so += o.so; self.sdif += o.sdif; self.n += o.n


# ==========================================================================
# PLOTAGEM (estilo relatorio)
# ==========================================================================
ESTILO = {"jaci": "-", "xc50": "--"}
MARCA = {"jaci": "o", "xc50": "s"}
COR_REG = {"Todo": "#1f77b4", "Continente": "#d62728", "Oceano": "#2ca02c",
           "Amazonia": "#1f77b4", "Nordeste": "#d62728", "Sudeste": "#2ca02c",
           "Sul": "#9467bd"}
_PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
        "#8c564b", "#000000"]


def estilo(m):
    return ESTILO.get(m, "-"), MARCA.get(m, "o")


def cores_limiar(limiares):
    import matplotlib.pyplot as plt
    if len(limiares) <= len(_PAL):
        return {t: _PAL[i] for i, t in enumerate(limiares)}
    cs = plt.cm.turbo(np.linspace(0.02, 0.98, len(limiares)))
    return {t: c for t, c in zip(limiares, cs)}


def _cor_modelo(modelos):
    base = {"jaci": "#1f77b4", "xc50": "#d62728"}
    return {m: base.get(m, _PAL[i % len(_PAL)]) for i, m in enumerate(modelos)}


def plota_continuas_regiao(dfc, regioes, unidade, titulo, arqsaida):
    """2x2: BIAS, MAE, RMSE, SCORR x prazo; cor=regiao, estilo=modelo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = dfc[dfc.regiao.isin(regioes)]
    if d.empty:
        return
    modelos = sorted(d.modelo.unique())
    metr = [("BIAS", f"BIAS ({unidade})"), ("MAE", f"MAE ({unidade})"),
            ("RMSE", f"RMSE ({unidade})"), ("SCORR", "SCORR")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, rot) in zip(axs.ravel(), metr):
        for rg in regioes:
            for m in modelos:
                sub = d[(d.regiao == rg) & (d.modelo == m)].sort_values("lead")
                if sub.empty or sub[col].isna().all():
                    continue
                ls, mk = estilo(m)
                ax.plot(sub.lead, sub[col], ls, marker=mk,
                        color=COR_REG.get(rg, "k"), markersize=5, lw=1.8)
        ax.set_title(rot); ax.set_xlabel("Prazo de previsao (dias)")
        ax.grid(alpha=0.3); ax.set_xticks(sorted(d.lead.unique()))
        if col == "BIAS":
            ax.axhline(0, color="grey", lw=0.8, ls=":")
    h = [Line2D([0], [0], color=COR_REG.get(r, "k"), lw=2.5, label=r) for r in regioes]
    h += [Line2D([0], [0], color="black", linestyle=estilo(m)[0],
                 marker=estilo(m)[1], label=m) for m in modelos]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.955),
               ncol=len(h), fontsize=9, framealpha=0.9, borderaxespad=0.3,
               columnspacing=1.0, handlelength=2.4)
    fig.suptitle(titulo, fontsize=13, x=0.5, y=0.995)
    fig.savefig(arqsaida, dpi=130); plt.close(fig); print(f"  {arqsaida}")


def plota_por_prazo(dfk, col, ylab, titulo, arqsaida, ref1=False):
    """Paineis por prazo (D+1..D+9): col x limiar; estilo=modelo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if dfk.empty:
        return
    modelos = sorted(dfk.modelo.unique())
    leads = sorted(dfk.lead.unique())[:9]
    n = len(leads); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                            squeeze=False)
    for k, ld in enumerate(leads):
        ax = axs.ravel()[k]
        for m in modelos:
            ls, mk = estilo(m)
            sub = dfk[(dfk.lead == ld) & (dfk.modelo == m)].sort_values("limiar_mm")
            if sub.empty:
                continue
            ax.plot(sub.limiar_mm, sub[col], ls, marker=mk, markersize=4,
                    lw=1.8, label=m)
        ax.set_title(f"D+{ld}"); ax.set_xlabel("Limiar (mm)"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        if ref1:
            ax.axhline(1, color="grey", lw=0.8, ls=":")
    for k in range(n, nrow * ncol):
        axs.ravel()[k].axis("off")
    h = [Line2D([0], [0], color="black", linestyle=estilo(m)[0],
                marker=estilo(m)[1], label=m) for m in modelos]
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.99),
               ncol=len(h), fontsize=10, framealpha=0.9, borderaxespad=0.3)
    fig.suptitle(titulo, fontsize=13, x=0.5, y=0.995)
    fig.savefig(arqsaida, dpi=130); plt.close(fig); print(f"  {arqsaida}")


def plota_mapas(arrs, grade_lats, grade_lons, unidade, titulo, arqsaida, difcmap="RdBu_r"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lats = np.asarray(grade_lats); lons = np.asarray(grade_lons)
    with np.errstate(invalid="ignore"):
        n = arrs["n"]
        pm = np.where(n > 0, arrs["sp"] / n, np.nan)
        om = np.where(n > 0, arrs["so"] / n, np.nan)
        bm = np.where(n > 0, arrs["sdif"] / n, np.nan)
    if lats[0] > lats[-1]:
        pm, om, bm = pm[::-1], om[::-1], bm[::-1]
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    val = np.concatenate([om[np.isfinite(om)], pm[np.isfinite(pm)]])
    vmin = float(np.nanpercentile(val, 1)) if val.size else 0.0
    vmax = float(np.nanpercentile(val, 99)) if val.size else 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    bv = np.abs(bm[np.isfinite(bm)])
    bmax = float(np.nanpercentile(bv, 99)) if bv.size else 1.0
    if not np.isfinite(bmax) or bmax <= 0:
        bmax = 1.0
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, campo, tit, cmap, vlim in [
        (ax[0], om, "Obs media (ref)", "viridis", (vmin, vmax)),
        (ax[1], pm, "Prev media", "viridis", (vmin, vmax)),
        (ax[2], bm, "Vies medio (Prev-Obs)", difcmap, (-bmax, bmax)),
    ]:
        im = a.imshow(campo, origin="lower", extent=ext, aspect="auto",
                      cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        a.set_title(tit); a.set_xlabel("lon"); a.set_ylabel("lat")
        fig.colorbar(im, ax=a, shrink=0.85, label=unidade)
    fig.suptitle(titulo); fig.tight_layout()
    fig.savefig(arqsaida, dpi=120); plt.close(fig); print(f"  {arqsaida}")


MESES = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
         7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}


def _soma_mapas(lista):
    """Soma sp/so/n de varios acumuladores de mapa (para o 'Todo periodo')."""
    sp = sum(np.asarray(a["sp"], float) for a in lista)
    so = sum(np.asarray(a["so"], float) for a in lista)
    n = sum(np.asarray(a["n"], float) for a in lista)
    return {"sp": sp, "so": so, "n": n}


def plota_mapas_mes(grade_lats, grade_lons, pormes, unidade, titulo, arqsaida,
                    difcmap="RdBu_r"):
    """Mapas de media diaria por periodo: colunas = [Todo, meses presentes];
    linhas = [Prev media, Vies medio (Prev-Obs)]. pormes: {mes_int: {sp,so,n}}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lats = np.asarray(grade_lats); lons = np.asarray(grade_lons)
    flip = lats[0] > lats[-1]
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    meses = sorted(pormes)
    cols = [("Todo periodo", _soma_mapas([pormes[m] for m in meses]))]
    cols += [(MESES.get(m, str(m)), pormes[m]) for m in meses]

    def med(a):
        with np.errstate(invalid="ignore"):
            n = np.asarray(a["n"], float)
            pm = np.where(n > 0, np.asarray(a["sp"], float) / n, np.nan)
            bm = np.where(n > 0, (np.asarray(a["sp"], float) -
                                  np.asarray(a["so"], float)) / n, np.nan)
        if flip:
            pm, bm = pm[::-1], bm[::-1]
        return pm, bm

    dados = [med(a) for _, a in cols]
    todas_p = np.concatenate([d[0][np.isfinite(d[0])] for d in dados]) if dados else np.array([])
    todas_b = np.concatenate([np.abs(d[1][np.isfinite(d[1])]) for d in dados]) if dados else np.array([])
    vmin = float(np.nanpercentile(todas_p, 1)) if todas_p.size else 0.0
    vmax = float(np.nanpercentile(todas_p, 99)) if todas_p.size else 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    bmax = float(np.nanpercentile(todas_b, 99)) if todas_b.size else 1.0
    if not np.isfinite(bmax) or bmax <= 0:
        bmax = 1.0

    nc = len(cols)
    fig, axs = plt.subplots(2, nc, figsize=(3.6 * nc + 1, 7.2), squeeze=False)
    for j, (rot, _a) in enumerate(cols):
        pm, bm = dados[j]
        im0 = axs[0, j].imshow(pm, origin="lower", extent=ext, aspect="auto",
                               cmap="viridis", vmin=vmin, vmax=vmax)
        axs[0, j].set_title(rot)
        im1 = axs[1, j].imshow(bm, origin="lower", extent=ext, aspect="auto",
                               cmap=difcmap, vmin=-bmax, vmax=bmax)
        for i in (0, 1):
            axs[i, j].set_xlabel("lon")
            if j == 0:
                axs[i, j].set_ylabel("lat")
    fig.colorbar(im0, ax=axs[0, :].tolist(), shrink=0.8, label=f"Prev media ({unidade})")
    fig.colorbar(im1, ax=axs[1, :].tolist(), shrink=0.8, label=f"Vies ({unidade})")
    fig.suptitle(titulo, fontsize=13)
    fig.savefig(arqsaida, dpi=120); plt.close(fig); print(f"  {arqsaida}")
