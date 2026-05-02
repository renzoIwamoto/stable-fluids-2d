"""
Operaciones del solver de fluidos.

Implementa los pasos del metodo Stable Fluids de Jos Stam:
- add_source : termino de fuerzas externas.
- diffuse    : viscosidad / difusion (implicito, Gauss-Seidel).
- advect     : auto-transporte (semi-Lagrangiano).
- project    : forzar incompresibilidad (Helmholtz-Hodge).
"""

import numpy as np


# ----------------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------------

def set_boundary(field: np.ndarray, kind: str = "scalar") -> None:
    """
    Aplica condiciones de borde a las ghost cells.

    kind:
        'scalar' : condicion de Neumann homogenea (copia el vecino).
                   Para density, presion, divergencia.
        'u'      : pared solida en bordes izquierdo/derecho.
                   La velocidad-x se invierte de signo en esas ghost cells
                   y se copia en bordes inferior/superior.
        'v'      : pared solida en bordes inferior/superior.
                   La velocidad-y se invierte de signo en esas ghost cells
                   y se copia en bordes izquierdo/derecho.
    """
    if kind == "scalar":
        field[0, 1:-1]  = field[1, 1:-1]
        field[-1, 1:-1] = field[-2, 1:-1]
        field[1:-1, 0]  = field[1:-1, 1]
        field[1:-1, -1] = field[1:-1, -2]
    elif kind == "u":
        # Pared en izquierda/derecha: invertimos componente x
        field[0, 1:-1]  = -field[1, 1:-1]
        field[-1, 1:-1] = -field[-2, 1:-1]
        # Arriba/abajo: copia (no hay pared que afecte u)
        field[1:-1, 0]  = field[1:-1, 1]
        field[1:-1, -1] = field[1:-1, -2]
    elif kind == "v":
        # Izquierda/derecha: copia
        field[0, 1:-1]  = field[1, 1:-1]
        field[-1, 1:-1] = field[-2, 1:-1]
        # Pared en abajo/arriba: invertimos componente y
        field[1:-1, 0]  = -field[1:-1, 1]
        field[1:-1, -1] = -field[1:-1, -2]
    else:
        raise ValueError(f"kind desconocido: {kind}")

    # Esquinas: promedio de los dos bordes adyacentes
    field[0, 0]   = 0.5 * (field[1, 0]   + field[0, 1])
    field[0, -1]  = 0.5 * (field[1, -1]  + field[0, -2])
    field[-1, 0]  = 0.5 * (field[-2, 0]  + field[-1, 1])
    field[-1, -1] = 0.5 * (field[-2, -1] + field[-1, -2])


# ----------------------------------------------------------------------
# Add source
# ----------------------------------------------------------------------

def add_source(field: np.ndarray, source: np.ndarray, dt: float) -> None:
    """field += source * dt (in-place)."""
    field += source * dt


# ----------------------------------------------------------------------
# Linear solver (Jacobi/Gauss-Seidel)
# ----------------------------------------------------------------------

def lin_solve(
    field: np.ndarray,
    field_prev: np.ndarray,
    a: float,
    c: float,
    kind: str,
    iterations: int = 20,
) -> None:
    """
    Resuelve field[i,j] * c = field_prev[i,j] + a * (suma vecinos).
    Iteracion vectorizada estilo Jacobi.
    """
    for _ in range(iterations):
        field[1:-1, 1:-1] = (
            field_prev[1:-1, 1:-1]
            + a * (
                field[0:-2, 1:-1]
                + field[2:,   1:-1]
                + field[1:-1, 0:-2]
                + field[1:-1, 2:  ]
            )
        ) / c
        set_boundary(field, kind)


# ----------------------------------------------------------------------
# Diffusion
# ----------------------------------------------------------------------

def diffuse(
    field: np.ndarray,
    field_prev: np.ndarray,
    diff: float,
    dt: float,
    kind: str,
    iterations: int = 20,
) -> None:
    """Difusion implicita."""
    N = field.shape[0] - 2
    a = diff * dt * N * N
    lin_solve(field, field_prev, a, 1 + 4 * a, kind, iterations)


# ----------------------------------------------------------------------
# Advection
# ----------------------------------------------------------------------

def advect(
    field: np.ndarray,
    field_prev: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    kind: str,
) -> None:
    """
    Adveccion semi-Lagrangiana: para cada celda interior, retrocede
    en el tiempo siguiendo (u, v) e interpola field_prev en esa
    posicion (interpolacion bilineal).
    """
    N = field.shape[0] - 2
    dt0 = dt * N

    i_idx, j_idx = np.meshgrid(
        np.arange(1, N + 1),
        np.arange(1, N + 1),
        indexing="ij",
    )

    x = i_idx - dt0 * u[1:-1, 1:-1]
    y = j_idx - dt0 * v[1:-1, 1:-1]

    x = np.clip(x, 0.5, N + 0.5)
    y = np.clip(y, 0.5, N + 0.5)

    i0 = np.floor(x).astype(np.int32)
    i1 = i0 + 1
    j0 = np.floor(y).astype(np.int32)
    j1 = j0 + 1

    s1 = x - i0
    s0 = 1.0 - s1
    t1 = y - j0
    t0 = 1.0 - t1

    field[1:-1, 1:-1] = (
        s0 * (t0 * field_prev[i0, j0] + t1 * field_prev[i0, j1])
        + s1 * (t0 * field_prev[i1, j0] + t1 * field_prev[i1, j1])
    )

    set_boundary(field, kind)


# ----------------------------------------------------------------------
# Projection (Helmholtz-Hodge)
# ----------------------------------------------------------------------

def project(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    div: np.ndarray,
    iterations: int = 20,
) -> None:
    """
    Proyecta el campo de velocidad (u, v) al subespacio de campos
    sin divergencia (incompresibles).

    Resuelve la ecuacion de Poisson para la presion p:
        laplacian(p) = divergencia(u, v)
    y luego corrige la velocidad:
        u -= grad_x(p)
        v -= grad_y(p)

    Parameters
    ----------
    u, v : np.ndarray
        Componentes del campo de velocidad. Modificados in-place.
    p, div : np.ndarray
        Buffers de trabajo del mismo shape que u, v.
        Se usan para presion y divergencia. Su contenido inicial
        se sobreescribe.
    iterations : int
        Iteraciones del solver de Poisson.
    """
    N = u.shape[0] - 2
    h = 1.0 / N  # espaciado fisico

    # 1. Calcular divergencia del campo de velocidad.
    div[1:-1, 1:-1] = -0.5 * h * (
        u[2:,   1:-1] - u[0:-2, 1:-1]
        + v[1:-1, 2:  ] - v[1:-1, 0:-2]
    )
    p[:, :] = 0.0
    set_boundary(div, "scalar")
    set_boundary(p, "scalar")

    # 2. Resolver Poisson: laplacian(p) = div.
    #    Forma estandar de Stam: a=1, c=4.
    lin_solve(p, div, a=1.0, c=4.0, kind="scalar", iterations=iterations)

    # 3. Restar el gradiente de p de la velocidad.
    u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[0:-2, 1:-1]) / h
    v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, 0:-2]) / h

    set_boundary(u, "u")
    set_boundary(v, "v")