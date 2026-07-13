#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leitor_binctl.py — leitor NATIVO de campos de superficie do Eta em formato
GrADS (.ctl/.bin), sem depender do GrADS instalado. Adaptado do projeto
"Script de verificacao" (read_eta_native). Usado quando a config declara
reader: binctl para uma variavel/modelo.

Expoe: campos_instantaneos(ctl_ou_template, spec, leads_horas, init)
  -> lista de (valido, lead_h, lead_d, nucleo.Campo)

Observacao: so le variaveis de SUPERFICIE (nlev 0/1). Para niveis (ex.: Z500),
use um .ctl/.bin ja no nivel desejado ou o caminho NetCDF.
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

import nucleo as N


def _parse_ctl(ctl_path):
    ctldir = os.path.dirname(os.path.abspath(ctl_path))
    info = {"vars": [], "options": ""}
    lines = open(ctl_path, encoding="latin-1").read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip(); low = line.lower()
        if low.startswith("dset"):
            d = line.split(None, 1)[1].strip()
            info["dset"] = os.path.join(ctldir, d[1:]) if d.startswith("^") else d
        elif low.startswith("options"):
            info["options"] += " " + low
        elif low.startswith("undef"):
            info["undef"] = float(line.split()[1])
        elif low.startswith("xdef"):
            p = line.split(); info.update(nx=int(p[1]), x0=float(p[3]), dx=float(p[4]))
        elif low.startswith("ydef"):
            p = line.split(); info.update(ny=int(p[1]), y0=float(p[3]), dy=float(p[4]))
        elif low.startswith("vars"):
            try:
                n = int(line.split()[1])
            except (IndexError, ValueError):
                n = 0
            for _ in range(n):
                i += 1
                if i >= len(lines):
                    break
                p = lines[i].split()
                if len(p) >= 2:
                    try:
                        info["vars"].append((p[0], int(p[1])))
                    except ValueError:
                        pass
        i += 1
    return info


def _tmpl_name(dset, vt):
    vt = pd.Timestamp(vt)
    return (dset.replace("%y4", f"{vt.year:04d}").replace("%y2", f"{vt.year % 100:02d}")
            .replace("%m2", f"{vt.month:02d}").replace("%d2", f"{vt.day:02d}")
            .replace("%h2", f"{vt.hour:02d}"))


def campos_instantaneos(ctl_path, spec, leads_horas, init):
    info = _parse_ctl(ctl_path)
    nx, ny = info["nx"], info["ny"]; fld = nx * ny
    undef = info.get("undef", -9.99e8)
    lat = info["y0"] + np.arange(ny) * info["dy"]
    lon = N.norm_lons(info["x0"] + np.arange(nx) * info["dx"])
    swap = ("byteswapped" in info["options"]) or ("big_endian" in info["options"])
    dtype = ">f4" if swap else "<f4"
    yrev = "yrev" in info["options"]

    order, idx = {}, 0
    for name, nlev in info["vars"]:
        order[name.lower()] = (idx, nlev); idx += max(1, nlev)
    var = (spec.get("var") or "").lower()
    if var not in order:
        raise RuntimeError(f"binctl: variavel '{var}' nao esta no ctl "
                           f"({[n for n, _ in info['vars']]})")
    off, nlev = order[var]
    if nlev not in (0, 1):
        raise RuntimeError(f"binctl: '{var}' nao e' de superficie (nlev={nlev})")

    unidade = spec.get("unidade"); para = spec.get("para")
    saida = []
    for lh in leads_horas:
        vt = pd.Timestamp(init) + timedelta(hours=int(lh))
        fn = _tmpl_name(info["dset"], vt)
        if not os.path.exists(fn):
            continue
        with open(fn, "rb") as f:
            f.seek(off * fld * 4)
            a = np.fromfile(f, dtype=dtype, count=fld)
        if a.size < fld:
            a = np.concatenate([a, np.full(fld - a.size, np.nan)])
        a = a.reshape(ny, nx).astype(float)
        a = np.where(a == undef, np.nan, a)
        if yrev:
            a = a[::-1, :]
        if unidade and para:
            a = N.converte_unidade(a, unidade, para)
        saida.append((vt, int(lh), int(round(lh / 24.0)),
                      N.Campo(a, lat, lon, nome=var)))
    return saida
