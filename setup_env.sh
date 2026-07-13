#!/bin/bash
# ==========================================================================
# setup_env.sh - cria (ou atualiza) o ambiente conda 'verificacao'.
#   ./setup_env.sh
# Depois:  conda activate verificacao
# ==========================================================================
set -e
cd "$(dirname "$0")"

ENVNAME=verificacao

# usa mamba se disponivel (bem mais rapido), senao conda
CONDA=conda
command -v mamba >/dev/null 2>&1 && CONDA=mamba

if conda env list | grep -qE "^\s*${ENVNAME}\s"; then
    echo ">> Ambiente '${ENVNAME}' ja existe: atualizando..."
    $CONDA env update -f environment.yml --prune
else
    echo ">> Criando ambiente '${ENVNAME}'..."
    $CONDA env create -f environment.yml
fi

echo ""
echo "Pronto. Ative com:"
echo "    conda activate ${ENVNAME}"
echo "Teste rapido:"
echo "    python verifica.py --config config_verificacao.yaml --max-rodadas 1"
