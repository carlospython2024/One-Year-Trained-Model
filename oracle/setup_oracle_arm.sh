#!/bin/bash
set -e

echo "🚀 Configurando entorno para Oracle Cloud ARM64"
echo "================================================"

# Detectar arquitectura
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]] && [[ "$ARCH" != "arm64" ]]; then
    echo "⚠️  WARNING: Esta máquina no es ARM64 (detectado: $ARCH)"
    echo "Este script está optimizado para Oracle Cloud ARM64"
    read -p "¿Continuar de todas formas? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "📦 Paso 1/5: Actualizando sistema..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

echo ""
echo "🔧 Paso 2/5: Instalando dependencias del sistema..."
sudo apt-get install -y -qq \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    gfortran \
    git \
    cmake \
    libomp-dev

echo ""
echo "🐍 Paso 3/5: Configurando entorno Python..."
python3 -m pip install --upgrade pip setuptools wheel --quiet

echo ""
echo "📚 Paso 4/5: Instalando dependencias Python..."
echo "   (Esto puede tardar varios minutos en ARM...)"

# Instalar en orden de dependencias para evitar problemas
pip3 install numpy --quiet
pip3 install pandas --quiet
pip3 install scipy --quiet
pip3 install scikit-learn --quiet

# ML libraries
pip3 install lightgbm --quiet
echo "   ✓ LightGBM instalado"

# Visualization
pip3 install matplotlib plotly seaborn --quiet
echo "   ✓ Librerías de visualización instaladas"

# Stats & Time Series
pip3 install statsmodels hmmlearn --quiet
echo "   ✓ Librerías estadísticas instaladas"

# Utilities
pip3 install python-dotenv requests pytz python-dateutil betacal --quiet
echo "   ✓ Utilidades instaladas"

# Try to install numba (may fail on some ARM systems)
echo ""
echo "   Intentando instalar Numba (opcional)..."
if pip3 install numba --quiet 2>/dev/null; then
    echo "   ✓ Numba instalado exitosamente"
else
    echo "   ⚠️  Numba no pudo instalarse (el código funcionará sin ella)"
fi

echo ""
echo "✅ Paso 5/5: Verificando instalaciones..."
python3 -c "import numpy; print(f'   ✓ NumPy {numpy.__version__}')"
python3 -c "import pandas; print(f'   ✓ Pandas {pandas.__version__}')"
python3 -c "import lightgbm; print(f'   ✓ LightGBM {lightgbm.__version__}')"
python3 -c "import sklearn; print(f'   ✓ scikit-learn {sklearn.__version__}')"

# Check optional
python3 -c "
try:
    import numba
    print(f'   ✓ Numba {numba.__version__}')
except:
    print('   ⚠️  Numba no disponible (opcional)')
" 2>/dev/null

echo ""
echo "================================================"
echo "✨ ¡Configuración completa!"
echo ""
echo "Próximos pasos:"
echo "  1. Configura tus credenciales en .env"
echo "  2. Ejecuta: python3 StepForwardTestingNewFeaturesmodelARM.py"
echo ""
