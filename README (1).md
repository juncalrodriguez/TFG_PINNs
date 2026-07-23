# PINNs para el modelado de la ecuación de la onda con la ecuación eikonal

Código y experimentos del Trabajo Fin de Grado **«PINNs para el modelado de la ecuación de la onda con la ecuación eikonal»**, en el que se resuelve la ecuación eikonal en medios homogéneos e isótropos sobre mallas tetraédricas tridimensionales mediante *Physics-Informed Neural Networks* (PINNs) implementadas en PyTorch, validando los resultados frente a la solución analítica exacta.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-informed-ee4c2c">
  <img alt="Licencia" src="https://img.shields.io/badge/Licencia-MIT-green">
</p>

> **Ficha académica**
> - **Titulación:** Grado en Matemáticas — Mención en Matemática Computacional
> - **Universidad:** Universidad Internacional de Valencia (VIU)
> - **Autora:** Juncal Rodríguez Palomo
> - **Tutor del TFT:** Jorge Patricio Sánchez Arciegas
> - **Curso académico:** 2025–2026 · Convocatoria: primera

---

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Fundamento matemático](#2-fundamento-matemático)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Requisitos e instalación](#4-requisitos-e-instalación)
5. [Datos de entrada: la malla](#5-datos-de-entrada-la-malla)
6. [Uso rápido](#6-uso-rápido)
7. [Scripts y parámetros](#7-scripts-y-parámetros)
8. [Reproducción de los experimentos](#8-reproducción-de-los-experimentos)
9. [Resultados de referencia](#9-resultados-de-referencia)
10. [Visualización con Meshalyzer](#10-visualización-con-meshalyzer)
11. [Limitaciones conocidas](#11-limitaciones-conocidas)
12. [Líneas de trabajo futuras](#12-líneas-de-trabajo-futuras)
13. [Cómo citar este trabajo](#13-cómo-citar-este-trabajo)
14. [Referencias bibliográficas](#14-referencias-bibliográficas)
15. [Licencia](#15-licencia)

---

## 1. Descripción del proyecto

La **ecuación eikonal** es una ecuación en derivadas parciales completamente no lineal de primer orden que describe el tiempo de llegada de un frente de onda a cada punto del dominio, sin necesidad de resolver la ecuación de ondas completa. Aparece de forma natural en óptica geométrica, acústica, sismología y electrofisiología cardíaca.

Este repositorio contiene la implementación que permite:

- **Calcular la solución analítica de referencia** de la ecuación eikonal en un medio homogéneo e isótropo sobre todos los nodos de una malla tetraédrica en formato openCARP.
- **Entrenar una PINN** que aproxima el campo de tiempos de llegada incorporando el residual de la ecuación eikonal directamente en la función de pérdida, mediante diferenciación automática y **sin utilizar datos etiquetados**.
- **Evaluar cuantitativamente** la aproximación obtenida (MAE, RMSE, error máximo, error mediano y percentiles del error relativo).
- **Exportar y visualizar** los campos escalares resultantes en formatos `.dat` y `.vtu`, compatibles con Meshalyzer y ParaView.

El trabajo estudia, además, la robustez de la metodología frente a cambios en los dos parámetros físicos fundamentales del problema: la **posición de la fuente de activación** y la **velocidad de propagación del medio**.

---

## 2. Fundamento matemático

### 2.1. La ecuación eikonal

Sobre un dominio $\Omega \subseteq \mathbb{R}^n$, la ecuación eikonal se escribe

$$\|\nabla T(x)\| = \frac{1}{v(x)}, \qquad x \in \Omega,$$

donde $T(x)$ es el tiempo de llegada (o función de fase) en el punto $x$ y $v(x)$ es la velocidad local de propagación. Las superficies de nivel $T(x) = \text{cte}$ son los frentes de onda y $\nabla T(x)$ es normal a ellas, señalando la dirección de propagación: las trayectorias que siguen esa dirección son los **rayos**.

Se trata de un caso particular de ecuación de Hamilton–Jacobi estacionaria, con Hamiltoniano $H(x,p) = v(x)\|p\| - 1$, y puede obtenerse también como aproximación asintótica de alta frecuencia (WKB) de la ecuación de ondas

$$\frac{\partial^2 u}{\partial t^2} = v^2(x)\,\nabla^2 u,$$

sustituyendo $u(x,t) = A(x)\,e^{i\omega(t - T(x))}$ y anulando el coeficiente del término dominante de orden $\omega^2$. El término de orden $\omega$ conduce a la **ecuación de transporte** $2\nabla A \cdot \nabla T + A\,\nabla^2 T = 0$, que gobierna la evolución de la amplitud a lo largo de los rayos.

### 2.2. Solución analítica en medio homogéneo

Si $v(x) \equiv v_0$ es constante y el frente parte de la fuente $x_s$ con $T(x_s) = 0$, la simetría radial del problema conduce a

$$T(x) = \frac{\|x - x_s\|}{v_0},$$

que en tres dimensiones se escribe

$$T(x,y,z) = \frac{\sqrt{(x-x_s)^2 + (y-y_s)^2 + (z-z_s)^2}}{v_0}.$$

Esta expresión es la **referencia exacta** con la que se validan todos los resultados de la PINN.

### 2.3. Formulación factorizada

El gradiente de $T$ es singular en la posición de la fuente, lo que desestabiliza tanto los métodos numéricos clásicos como el entrenamiento de la red. Para evitarlo se emplea la **factorización**

$$T(x) = T_0(x)\,\tau(x), \qquad T_0(x) = \frac{\|x - x_s\|}{v(x_s)}, \qquad \tau(x_s) = 1,$$

que traslada el comportamiento singular a la función conocida $T_0$ y deja para la red una función correctora $\tau$ mucho más regular. Sustituyendo en la forma cuadrática de la ecuación se obtiene

$$\|\nabla T_0\|^2 \tau^2 + T_0^2 \|\nabla \tau\|^2 + 2\,T_0\,\tau\,\big(\nabla T_0 \cdot \nabla \tau\big) = \frac{1}{v^2(x)}.$$

En la implementación esta idea se materializa como

$$T(x) = d(x)\,N(x),$$

donde $d(x) = \|x - x_s\|$ es la distancia euclídea a la fuente (calculada en **coordenadas físicas**, en mm) y $N(x)$ es la salida de la red (evaluada sobre **coordenadas normalizadas** en $[-1,1]$). Con ello la condición $T(x_s) = 0$ se satisface **exactamente por construcción**, sin necesidad de un término de pérdida de contorno.

> **Nota de diseño.** Preservar la distancia en unidades físicas mantiene el significado dimensional del gradiente, mientras que normalizar la entrada de la red mejora el condicionamiento del problema de optimización. Ambas representaciones conviven deliberadamente en el `forward` de la clase `DistancePINN`.

### 2.4. Función de pérdida

Se define el residual físico

$$R(x) = \|\nabla T(x)\| - \frac{1}{v},$$

y la pérdida como su error cuadrático medio sobre los puntos de colocación:

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} = \frac{1}{N}\sum_{i=1}^{N}\left(\|\nabla T(x_i)\| - \frac{1}{v}\right)^2.$$

Las derivadas espaciales se obtienen por **diferenciación automática** (`torch.autograd.grad`) respecto de las coordenadas físicas de entrada, evitando cualquier aproximación por diferencias finitas. Se aplica una **máscara esférica de exclusión de radio 0,5 mm** alrededor de la fuente para no evaluar el residual en el entorno inmediato de la singularidad: un radio suficientemente pequeño para no alterar la solución física y suficientemente grande para evitar inestabilidades numéricas.

### 2.5. Inicialización informada por la física

El sesgo de la última capa se inicializa con la **lentitud** del medio, $s = 1/v$, de modo que la red arranca en

$$T(x) \approx \frac{d(x)}{v},$$

es decir, en la solución exacta del caso homogéneo. Esto reduce considerablemente el número de iteraciones necesarias para alcanzar una solución precisa.

---

## 3. Estructura del repositorio

```
TFG_PINNs/
├── README.md
│
├── meshes/                              # Malla tetraédrica en formato openCARP
│   ├── block.pts                        # Coordenadas de los nodos
│   ├── block.elem                       # Conectividad tetraédrica
│   ├── block.lon                        # Información direccional del medio
│   └── block.surf                       # Superficies de la malla
│
└── scripts/
    ├── eikonal_analitycal_solution.py   # ★ Solución analítica de referencia
    └── pinn/
        ├── pinn_train_bueno2.py         # ★ Entrenamiento de la PINN (versión definitiva)
        ├── pinnpinn_train.py            # Versión preliminar
        └── pinn_eikonal_opencarp_*.py   # Versiones preliminares
```

Los **dos scripts marcados con ★** son los utilizados para obtener todos los resultados presentados en la memoria:

| Script | Función |
|---|---|
| `scripts/pinn/pinn_train_bueno2.py` | Versión definitiva del código de entrenamiento. Implementa la arquitectura final de la PINN (clase `DistancePINN`), la función de pérdida basada en el residual eikonal, el bucle de optimización y la exportación de resultados. |
| `scripts/eikonal_analitycal_solution.py` | Calcula la solución analítica de la ecuación eikonal sobre la malla. Incorpora soporte para múltiples fuentes de activación, medios anisótropos y exportación automática a `.dat` y `.vtu`. |

El resto de ficheros de `scripts/pinn/` corresponden a **versiones preliminares** en las que se detectaron y corrigieron errores hasta alcanzar la versión definitiva. Se conservan por trazabilidad del desarrollo y **no deben utilizarse para reproducir los resultados**.

---

## 4. Requisitos e instalación

### 4.1. Dependencias

| Biblioteca | Uso |
|---|---|
| **NumPy** | Gestión de vectores y matrices, operaciones numéricas |
| **PyTorch** | Construcción de la red, diferenciación automática y entrenamiento |
| **MeshIO** | Lectura y escritura de mallas tridimensionales y resultados numéricos |
| **Matplotlib** | Validación y análisis gráfico de resultados |
| *(externo)* **Meshalyzer / openCARP** | Visualización 3D de los campos de tiempos |

### 4.2. Instalación

```bash
# Clonar el repositorio
git clone https://github.com/juncalrodriguez/TFG_PINNs.git
cd TFG_PINNs

# Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Instalar dependencias
pip install numpy torch meshio matplotlib
```

> **Sobre la GPU.** El entrenamiento funciona en CPU, pero es notablemente más rápido con CUDA. Si se dispone de GPU NVIDIA, conviene instalar PyTorch siguiendo las instrucciones oficiales de <https://pytorch.org/get-started/locally/>.

---

## 5. Datos de entrada: la malla

Los experimentos se realizan sobre un dominio tridimensional discretizado mediante una **malla tetraédrica en formato openCARP**, compuesta por **112 211 nodos**. Este tipo de malla es adecuado para representar geometrías complejas y es habitual en simulaciones de propagación de señales y frentes de onda.

| Archivo | Contenido |
|---|---|
| `meshes/block.pts` | Coordenadas tridimensionales de los nodos |
| `meshes/block.elem` | Conectividad entre nodos mediante tetraedros |
| `meshes/block.lon` | Información direccional del medio (anisotropía) |
| `meshes/block.surf` | Superficies de la malla |

**Unidades.** Las coordenadas de `block.pts` están expresadas originalmente en **micrómetros** y se convierten internamente a **milímetros** mediante un factor de escala de `0.001`. Todos los cálculos posteriores (distancias, gradientes y tiempos de llegada) se realizan en **mm** y **ms**, de forma que las velocidades quedan expresadas en **mm/ms** y las posiciones de las fuentes en **mm**.

Cada nodo queda representado por sus coordenadas espaciales $x = (x,y,z) \in \Omega \subset \mathbb{R}^3$, y sobre ese mismo conjunto de puntos se evalúan tanto la solución analítica como las predicciones de la red, lo que permite compararlas de forma cuantitativa y visual.

---

## 6. Uso rápido

Flujo completo para el **caso base** (fuente en el origen, $v = 0{,}5$ mm/ms), ejecutado desde la raíz del repositorio:

```bash
mkdir -p out

# 1) Solución analítica de referencia
python scripts/eikonal_analitycal_solution.py \
    --meshbase meshes/block.pts \
    --src 0 0 0 \
    --vl 500 \
    --outdir out/T_iso_nodes.dat

# 2) Entrenamiento de la PINN
python scripts/pinn/pinn_train_bueno2.py \
    --basename meshes/block \
    --source 0 0 0 \
    --epochs 15000 \
    --cv 0.5 \
    --out-vtu out/T_pinn_bueno2.vtu
```

El paso 3, el cálculo de las métricas de error, se detalla en la [sección 8](#83-cálculo-de-las-métricas-de-error).

---

## 7. Scripts y parámetros

### 7.1. `scripts/eikonal_analitycal_solution.py`

Calcula $T(x_i) = \|x_i - x_s\| / v$ en cada nodo de la malla y exporta el campo resultante.

| Argumento | Descripción |
|---|---|
| `--meshbase` | Ruta al archivo `.pts` de la malla |
| `--src` | Coordenadas $(x_s, y_s, z_s)$ de la fuente de activación |
| `--vl` | Velocidad de propagación (véase la tabla de equivalencias más abajo) |
| `--outdir` | Ruta de salida para el campo de tiempos |

**Salidas:** campo escalar nodal en `.dat` (un valor por línea, en el orden de los nodos) y geometría con el campo asociado en `.vtu`.

### 7.2. `scripts/pinn/pinn_train_bueno2.py`

Define y entrena la PINN y exporta el campo de tiempos predicho.

| Argumento | Descripción |
|---|---|
| `--basename` | Prefijo común de los archivos de malla, sin extensión |
| `--source` | Coordenadas de la fuente de activación, en mm |
| `--epochs` | Número de épocas de entrenamiento |
| `--cv` | Velocidad de propagación, en mm/ms |
| `--out-vtu` | Fichero `.vtu` de salida |

### 7.3. Equivalencia entre los parámetros de velocidad

Los dos scripts reciben la velocidad en escalas distintas. Esta es la correspondencia utilizada en todos los experimentos de la memoria:

| Velocidad física | `--vl` (solución analítica) | `--cv` (PINN) |
|---|---|---|
| 0,3 mm/ms | `300` | `0.3` |
| 0,5 mm/ms | `500` | `0.5` |
| 0,8 mm/ms | `800` | `0.8` |
| 1,0 mm/ms | `1000` | `1.0` |

> ⚙️ Conviene tenerlo presente al comparar campos: un mismo caso físico se especifica con valores numéricos diferentes en cada script.

### 7.4. Configuración de la PINN

| Parámetro | Valor |
|---|---|
| Dimensión de entrada | 3 (coordenadas $x, y, z$ normalizadas) |
| Número de capas ocultas | 4 |
| Neuronas por capa | 64 |
| Arquitectura completa | `[3, 64, 64, 64, 64, 1]` |
| Función de activación | `tanh` |
| Dimensión de salida | 1 |
| Optimizador | Adam |
| Tasa de aprendizaje | $10^{-3}$ |
| Número de épocas | 15 000 |
| Radio de exclusión en torno a la fuente | 0,5 mm |
| Inicialización del sesgo de salida | $1/v$ (lentitud del medio) |
| Velocidad de propagación | Variable (parámetro del experimento) |

La activación `tanh` se elige por su **suavidad y derivabilidad**, propiedades imprescindibles cuando hay que calcular gradientes espaciales por diferenciación automática dentro de la propia función de pérdida. La profundidad de la red busca un equilibrio entre capacidad de representación y coste computacional.

---

## 8. Reproducción de los experimentos

Todos los experimentos comparten arquitectura, optimizador y número de épocas, de modo que las diferencias observadas se deben **exclusivamente** a los parámetros físicos modificados. Los comandos se muestran con rutas relativas a la raíz del repositorio.

### 8.1. Experimento 1 — Caso base

Fuente en el origen, $v = 0{,}5$ mm/ms. Valida la capacidad de la PINN para reproducir la solución exacta.

```bash
python scripts/eikonal_analitycal_solution.py \
    --meshbase meshes/block.pts --src 0 0 0 --vl 500 \
    --outdir out/T_iso_nodes.dat

python scripts/pinn/pinn_train_bueno2.py \
    --basename meshes/block --source 0 0 0 --epochs 15000 \
    --cv 0.5 --out-vtu out/T_pinn_bueno2.vtu
```

### 8.2. Experimento 2 — Cambio en la posición de la fuente

Fuente desplazada a una esquina del dominio, con la misma velocidad. La distribución de tiempos deja de ser aproximadamente simétrica.

```bash
python scripts/eikonal_analitycal_solution.py \
    --meshbase meshes/block.pts --src -50 -50 -5 --vl 500 \
    --outdir out/T_iso_esquina.dat

python scripts/pinn/pinn_train_bueno2.py \
    --basename meshes/block --source -50 -50 -5 --epochs 15000 \
    --cv 0.5 --out-vtu out/T_pinn_esquina.vtu
```

### 8.3. Experimento 3 — Cambio en la velocidad de propagación

Fuente fija en el origen y velocidades $v \in \{0{,}3,\ 0{,}8,\ 1{,}0\}$ mm/ms.

```bash
# Soluciones analíticas
for V in 300 800 1000; do
  python scripts/eikonal_analitycal_solution.py \
      --meshbase meshes/block.pts --src 0 0 0 --vl $V \
      --outdir out/T_iso_v${V}.dat
done

# Entrenamientos de la PINN
for CV in 0.3 0.8 1.0; do
  python scripts/pinn/pinn_train_bueno2.py \
      --basename meshes/block --source 0 0 0 --epochs 15000 \
      --cv $CV --out-vtu out/T_pinn_v${CV}.vtu
done
```

> **Nota.** El caso $v = 1{,}5$ mm/ms se planteó inicialmente pero **no pudo completarse** con los recursos hardware disponibles, debido al elevado coste computacional del entrenamiento. La limitación es exclusivamente computacional, no metodológica: el procedimiento es aplicable a velocidades mayores si se dispone de más capacidad de cálculo.

### 8.4. Cálculo de las métricas de error

Sea $T_{\text{ana}}$ la solución analítica y $T_{\text{pinn}}$ la aproximación de la red. El error en cada nodo es $e_i = T_{\text{pinn}}(x_i) - T_{\text{ana}}(x_i)$, y a partir de él se calculan:

$$\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}|e_i|, \qquad
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}, \qquad
\mathrm{MAX} = \max_i |e_i|.$$

El MAE se expresa en las mismas unidades que la variable estudiada, lo que facilita su interpretación; el RMSE penaliza más los errores grandes y resulta útil para detectar regiones con desviaciones significativas.

El siguiente fragmento reproduce la evaluación realizada en la memoria:

```python
import numpy as np

Tp = np.loadtxt("out/T_pinn_bueno2.dat").reshape(-1)   # PINN
Ta = np.loadtxt("out/T_iso_nodes.dat").reshape(-1)     # Analítica

if Tp.shape[0] != Ta.shape[0]:
    raise ValueError(
        f"Longitudes distintas: PINN={Tp.shape[0]} vs Analítica={Ta.shape[0]}. "
        "Asegúrate de que ambos campos son nodales (mismos nodos y mismo orden)."
    )

err     = Tp - Ta
abs_err = np.abs(err)
rel_err = abs_err / np.maximum(np.abs(Ta), 1e-12)

print(f"MAE     = {abs_err.mean():.6e}")
print(f"RMSE    = {np.sqrt((err**2).mean()):.6e}")
print(f"MAX|e|  = {abs_err.max():.6e}")
print(f"MED|e|  = {np.median(abs_err):.6e}")
print(f"MRE     = {rel_err.mean():.6e}")
print(f"MED rel = {np.median(rel_err):.6e}")
print(f"P95 rel = {np.quantile(rel_err, 0.95):.6e}")
```

---

## 9. Resultados de referencia

### 9.1. Caso base — fuente $(0,0,0)$, $v = 0{,}5$ mm/ms

| Métrica | Valor |
|---|---|
| MAE | $8{,}31 \times 10^{-5}$ |
| RMSE | $1{,}30 \times 10^{-4}$ |
| Error máximo | $1{,}30 \times 10^{-3}$ |
| Error mediano | $5{,}24 \times 10^{-5}$ |
| P95 del error relativo | $2{,}62 \times 10^{-6}$ |

El error mediano, inferior al MAE, indica que más de la mitad de los nodos presentan errores todavía menores que el valor medio. El percentil 95 del error relativo, del orden de $10^{-6}$, muestra que el 95 % de los nodos presenta discrepancias muy pequeñas respecto de la solución exacta.

### 9.2. Efecto de la posición de la fuente ($v = 0{,}5$ mm/ms)

| Fuente | MAE | RMSE | Error máximo |
|---|---|---|---|
| $(0, 0, 0)$ | $8{,}31 \times 10^{-5}$ | $1{,}30 \times 10^{-4}$ | $1{,}30 \times 10^{-3}$ |
| $(-50, -50, -5)$ | $4{,}20 \times 10^{-4}$ | $6{,}51 \times 10^{-4}$ | $5{,}23 \times 10^{-3}$ |

Al desplazar la fuente a una esquina del dominio, la red debe aproximar un campo de tiempos más complejo, lo que explica el aumento de aproximadamente un orden de magnitud en todas las métricas. Aun así, los errores siguen siendo muy pequeños frente a la escala característica de los tiempos de propagación, y las mayores discrepancias permanecen localizadas.

### 9.3. Efecto de la velocidad de propagación (fuente en el origen)

| Velocidad (mm/ms) | MAE | RMSE | Error máximo |
|---|---|---|---|
| 0,3 | $8{,}49 \times 10^{-5}$ | $1{,}31 \times 10^{-4}$ | $1{,}36 \times 10^{-3}$ |
| 0,5 | $8{,}31 \times 10^{-5}$ | $1{,}30 \times 10^{-4}$ | $1{,}30 \times 10^{-3}$ |
| 0,8 | $8{,}23 \times 10^{-5}$ | $1{,}29 \times 10^{-4}$ | $1{,}02 \times 10^{-3}$ |
| 1,0 | $1{,}39 \times 10^{-4}$ | $2{,}13 \times 10^{-4}$ | $1{,}92 \times 10^{-3}$ |

Las métricas permanecen prácticamente constantes al variar la velocidad: la precisión del modelo **no depende de un valor concreto de este parámetro**. Un cambio de velocidad modifica la escala temporal de la propagación, pero no la geometría de los frentes de onda.

### 9.4. Sobre la convergencia

La función de pérdida desciende muy rápidamente en las primeras épocas y continúa disminuyendo hasta estabilizarse en valores muy reducidos. Aparecen **picos puntuales** durante el entrenamiento, habituales con el optimizador Adam por el carácter adaptativo de las actualizaciones de los parámetros; no afectan a la tendencia global, claramente decreciente.

> **Nota del desarrollo.** En las primeras implementaciones, **sin la formulación factorizada**, la función de pérdida se mantenía prácticamente constante a lo largo del entrenamiento: la red no convergía. La adopción de $T(x) = d(x)\,N(x)$ fue el cambio decisivo que estabilizó el proceso de optimización y mejoró significativamente la precisión de las soluciones obtenidas.

---

## 10. Visualización con Meshalyzer

Las representaciones gráficas del campo de tiempos se generan con **Meshalyzer**, integrado en openCARP.

1. Colocar el ejecutable de Meshalyzer en la **misma carpeta** que los archivos generados por la simulación: la geometría de la malla y el campo de tiempos calculado.
2. Ejecutar la aplicación.
3. Abrir la malla mediante `File → Read Data`.
4. Ajustar la escala de colores con el botón `colour scale`.

Para que la comparación entre configuraciones sea significativa conviene **utilizar el mismo rango de colores** en todas las capturas, y mantener visible el panel lateral de Meshalyzer, que muestra la escala y los valores mínimo y máximo del campo representado.

**Validación cualitativa.** En un medio homogéneo deben observarse frentes de onda aproximadamente esféricos centrados en la fuente, con una transición de color suave y continua. Cualquier desviación apreciable de este patrón (discontinuidades u oscilaciones numéricas) apuntaría a un problema en el entrenamiento o en la definición del modelo: la inspección visual funciona, por tanto, como mecanismo de validación adicional a las métricas numéricas.

Los ficheros `.vtu` pueden abrirse también directamente con **ParaView**.

---

## 11. Limitaciones conocidas

Se documentan de forma explícita para delimitar el alcance de las conclusiones:

- **Medio homogéneo e isótropo.** Todos los experimentos suponen velocidad constante en todo el dominio. Aunque la implementación gestiona mallas tridimensionales, el entrenamiento **no** se ha extendido a medios heterogéneos o anisótropos, donde la velocidad depende de la posición o de la dirección de propagación.
- **Validación frente a solución exacta.** La precisión se cuantifica comparando con una solución analítica conocida. No se garantiza el mismo comportamiento en problemas sin solución exacta disponible, donde habría que recurrir a métodos numéricos de referencia.
- **Generalización mediante reentrenamiento.** Cada configuración física requiere un entrenamiento independiente. **No se trata de una PINN paramétrica**: un único modelo entrenado no resuelve simultáneamente distintas velocidades o posiciones de la fuente. La «generalización» analizada debe entenderse como la capacidad de la *metodología* para adaptarse a nuevos escenarios.
- **Arquitectura fija.** No se ha realizado un estudio sistemático sobre el número de capas, el número de neuronas, las funciones de activación o los hiperparámetros de entrenamiento. Podrían existir configuraciones que mejorasen la precisión o redujeran el coste computacional.
- **Factorización específica.** El comportamiento de la factorización basada en la distancia euclídea no se ha estudiado para configuraciones más generales ni para otras formulaciones de la ecuación eikonal.
- **Coste computacional.** El entrenamiento sigue siendo un proceso costoso; esta limitación impidió completar el caso $v = 1{,}5$ mm/ms.

---

## 12. Líneas de trabajo futuras

- Extensión a **medios heterogéneos y anisótropos**, con $v = v(x)$ y tensores de conductividad.
- Incorporación de **múltiples fuentes de activación** simultáneas.
- Desarrollo de **PINNs paramétricas** capaces de resolver distintas configuraciones físicas mediante un único entrenamiento.
- Estudio sistemático de **arquitecturas más profundas o adaptativas** y de sus hiperparámetros.
- **Estrategias adaptativas de muestreo** de los puntos de colocación y técnicas avanzadas de optimización.
- Aplicación sobre **geometrías cardíacas realistas** obtenidas a partir de imágenes médicas.

---

## 13. Cómo citar este trabajo

```bibtex
@thesis{RodriguezPalomo2026PINNsEikonal,
  author = {Rodríguez Palomo, Juncal},
  title  = {PINNs para el modelado de la ecuación de la onda con la ecuación eikonal},
  type   = {Trabajo Fin de Grado},
  school = {Universidad Internacional de Valencia (VIU)},
  address = {Valencia, España},
  year   = {2026},
  note   = {Grado en Matemáticas. Director: Jorge Patricio Sánchez Arciegas},
  url    = {https://github.com/juncalrodriguez/TFG_PINNs}
}
```

**Palabras clave:** ecuación eikonal, ecuación de la onda, *Physics-Informed Neural Networks*, aprendizaje profundo, ecuaciones en derivadas parciales, propagación de ondas, diferenciación automática, PyTorch.

---

## 14. Referencias bibliográficas

- Goodfellow, I., Bengio, Y. y Courville, A. (2016). *Deep Learning*. MIT Press.
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S. y Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3, 422–440.
- Lagaris, I. E., Likas, A. y Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000.
- Plaza, R. G. (2020). *Ecuaciones diferenciales parciales. Lección 1.5: Ejemplos. La ecuación de la eikonal*. Material docente.
- Plaza, R. G. (2023). *Ecuaciones diferenciales parciales: ecuaciones de la eikonal y de Hamilton–Jacobi (sección 1)*. Material docente.
- PyTorch Contributors (2025). *PyTorch*. <https://pytorch.org>
- Raissi, M., Perdikaris, P. y Karniadakis, G. E. (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Sethian, J. A. (1999). *Level Set Methods and Fast Marching Methods*. Cambridge University Press.

---

## 15. Licencia

Este proyecto se distribuye bajo la **licencia MIT**.

```
MIT License

Copyright (c) 2026 Juncal Rodríguez Palomo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**Autora:** Juncal Rodríguez Palomo
**Tutor del TFT:** Jorge Patricio Sánchez Arciegas
**Universidad Internacional de Valencia (VIU)** — Grado en Matemáticas, curso 2025–2026
