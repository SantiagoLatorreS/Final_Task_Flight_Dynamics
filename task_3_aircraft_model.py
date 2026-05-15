import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import warnings
warnings.filterwarnings('ignore')

# CONSTANTES GLOBALES
G = 9.81
RHO = 1.225  # kg/m³

# FUNCIÓN XDOT - MODELO RCAM
def xdot(X, U, rho=RHO):

    X = np.asarray(X, dtype=float).flatten()
    U = np.asarray(U, dtype=float).flatten()

    # --- Extraer estados ---
    u, v, w, p, q, r, phi, theta, psi = X

    # --- Constantes del avión (RCAM GARTEUR) ---
    m = 120000.0       # kg - masa total
    cbar = 6.6         # m - cuerda media aerodinámica
    lt = 24.8          # m - distancia cola-CG
    S = 260.0          # m² - área alar
    St = 64.0          # m² - área cola

    # Centro de gravedad
    Xcg = 0.23 * cbar
    Ycg = 0.0
    Zcg = 0.10 * cbar

    # Centro aerodinámico
    Xac = 0.12 * cbar
    Yac = 0.0
    Zac = 0.0

    # Motores
    Xapt1, Yapt1, Zapt1 = 0.0, -7.94, -1.9   # Motor 1
    Xapt2, Yapt2, Zapt2 = 0.0,  7.94, -1.9   # Motor 2

    g = G
    deg2rad = np.pi / 180.0

    # Step 1: Saturar controles
    u1 = np.clip(U[0], -25*deg2rad, 25*deg2rad)    # Alerón: ±25°
    u2 = np.clip(U[1], -25*deg2rad, 10*deg2rad)     # Estabilizador: -25° a +10°
    u3 = np.clip(U[2], -30*deg2rad, 30*deg2rad)     # Timón: ±30°
    u4 = np.clip(U[3], 0.0, 1.0)                    # Throttle 1: 0 a 1 
    u5 = np.clip(U[4], 0.0, 1.0)                    # Throttle 2: 0 a 1

    # Step 2: Variables aerodinámicas intermedias
    Va = max(np.sqrt(u**2 + v**2 + w**2), 1e-6)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(np.clip(v / Va, -1.0, 1.0))
    Q = 0.5 * rho * Va**2

    wbe_b = np.array([p, q, r])
    V_b = np.array([u, v, w])

    # Step 3: Coeficientes aerodinámicos
    # Parámetros de sustentación
    alpha_L0 = -11.5 * deg2rad
    n = 5.5
    alpha_switch = 14.5 * deg2rad
    a3, a2, a1, a0 = -768.5, 609.2, -155.2, 15.212
    depsda = 0.25

    # CL wing-body
    if alpha <= alpha_switch:
        CL_wb = n * (alpha - alpha_L0)
    else:
        CL_wb = a3 * alpha**3 + a2 * alpha**2 + a1 * alpha + a0

    # CL cola
    epsilon = depsda * (alpha - alpha_L0)
    alpha_t = alpha - epsilon + u2 + 1.3 * q * lt / Va
    CL_t = 3.1 * (St / S) * alpha_t

    # Coeficientes totales
    CL = CL_wb + CL_t
    CD = 0.13 + 0.07 * (n * alpha + 0.654)**2
    CY = -1.6 * beta + 0.24 * u3

    # Step 4: Fuerzas aerodinámicas en ejes estabilidad
    FA_s = np.array([-CD * Q * S, CY * Q * S, -CL * Q * S])

    # Rotación de ejes estabilidad a ejes cuerpo
    C_bs = np.array([
        [np.cos(alpha), 0.0, -np.sin(alpha)],
        [0.0, 1.0, 0.0],
        [np.sin(alpha), 0.0,  np.cos(alpha)]
    ])
    FA_b = C_bs @ FA_s

    # Step 5-6: Momentos aerodinámicos
    eta = np.array([
        -1.4 * beta,
        -0.59 - (3.1 * (St * lt) / (S * cbar)) * (alpha - epsilon),
        (1.0 - alpha * (180.0 / (15.0 * np.pi))) * beta
    ])

    dCMdx = (cbar / Va) * np.array([
        [-11.0, 0.0, 5.0],
        [0.0, (-4.03 * (St * lt**2) / (S * cbar**2)), 0.0],
        [1.7, 0.0, -11.5]
    ])

    dCMdu = np.array([
        [-0.6, 0.0, 0.22],
        [0.0, (-3.1 * (St * lt) / (S * cbar)), 0.0],
        [0.0, 0.0, -0.63]
    ])

    CMac_b = eta + dCMdx @ wbe_b + dCMdu @ np.array([u1, u2, u3])
    MAac_b = CMac_b * Q * S * cbar

    # Transferir momento del AC al CG
    rcg_b = np.array([Xcg, Ycg, Zcg])
    rac_b = np.array([Xac, Yac, Zac])
    MAcg_b = MAac_b + np.cross(FA_b, rcg_b - rac_b)

    # Step 7: Propulsión
    F1 = u4 * m * g
    F2 = u5 * m * g

    FE1_b = np.array([F1, 0.0, 0.0])
    FE2_b = np.array([F2, 0.0, 0.0])
    FE_b = FE1_b + FE2_b

    mew1 = np.array([Xcg - Xapt1, Yapt1 - Ycg, Zcg - Zapt1])
    mew2 = np.array([Xcg - Xapt2, Yapt2 - Ycg, Zcg - Zapt2])
    MEcg_b = np.cross(mew1, FE1_b) + np.cross(mew2, FE2_b)

    # Step 8: Gravedad
    g_b = np.array([
        -g * np.sin(theta),
         g * np.cos(theta) * np.sin(phi),
         g * np.cos(theta) * np.cos(phi)
    ])
    Fg_b = m * g_b

    # Step 9: Ecuaciones de fuerza
    F_b = Fg_b + FE_b + FA_b
    x012_dot = (1.0 / m) * F_b - np.cross(wbe_b, V_b)

    # Step 10: Ecuaciones de momento
    Ib = m * np.array([
        [40.07,  0.0,    -2.0923],
        [0.0,    64.0,    0.0],
        [-2.0923, 0.0,   99.92]
    ])
    invIb = np.linalg.inv(Ib)

    Mcg_b = MAcg_b + MEcg_b
    x345_dot = invIb @ (Mcg_b - np.cross(wbe_b, Ib @ wbe_b))

    # Cinemática de Euler
    H_phi = np.array([
        [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0.0, np.cos(phi),                 -np.sin(phi)],
        [0.0, np.sin(phi) / np.cos(theta),  np.cos(phi) / np.cos(theta)]
    ])
    x678_dot = H_phi @ wbe_b

    return np.concatenate([x012_dot, x345_dot, x678_dot])


# INTEGRADOR RK4 CON PASO DISCRETO

def rk4_step(X, U, dt_internal, rho=RHO):
    k1 = xdot(X, U, rho)
    k2 = xdot(X + 0.5 * dt_internal * k1, U, rho)
    k3 = xdot(X + 0.5 * dt_internal * k2, U, rho)
    k4 = xdot(X + dt_internal * k3, U, rho)
    return X + (dt_internal / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(x0, t_span, control_func, dt=1.0, dt_internal=0.01, rho=RHO):
    N = len(t_span)
    sol = np.zeros((N, 9))
    sol[0] = x0

    n_substeps = int(dt / dt_internal)

    for i in range(1, N):
        t = t_span[i-1]
        U = control_func(t)
        X_current = sol[i-1].copy()
        # Integrar n_substeps veces con dt_internal
        for _ in range(n_substeps):
            X_current = rk4_step(X_current, U, dt_internal, rho)
        sol[i] = X_current

    return sol


# FUNCIONES DE CONTROL PARA CADA CASO

def controls_case1(t):
    return np.array([0.0, -0.1, 0.0, 0.08, 0.08])

def controls_case2(t):
    delta_a = np.radians(5.0) if 30 <= t <= 32 else 0.0
    return np.array([delta_a, -0.1, 0.0, 0.08, 0.08])

def controls_case3(t):
    throttle1 = 0.0 if t >= 30 else 0.08
    return np.array([0.0, -0.1, 0.0, throttle1, 0.08])


# GRÁFICAS

LABELS = ['u (m/s)', 'v (m/s)', 'w (m/s)',
          'p (rad/s)', 'q (rad/s)', 'r (rad/s)',
          'φ (rad)', 'θ (rad)', 'ψ (rad)']

DARK_BG  = '#1E1A18'
DARK_AX  = '#2C2520'
GRID_CLR = '#A67C52'
TXT_CLR  = '#E6C280'


def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor(DARK_AX)
    ax.set_xlabel(xlabel, color=TXT_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=TXT_CLR, fontsize=9)
    if title:
        ax.set_title(title, color=TXT_CLR, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TXT_CLR, labelsize=8)
    ax.grid(True, color=GRID_CLR, alpha=0.4, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)


def plot_case(t, sol, case_title, color='#E6C280', highlights=None):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), facecolor=DARK_BG)
    fig.suptitle(case_title, fontsize=15, fontweight='bold', color=TXT_CLR)

    for i in range(9):
        ax = axes[i // 3, i % 3]
        style_ax(ax, 'Tiempo (s)', LABELS[i])
        ax.plot(t, sol[:, i], color=color, linewidth=1.8)
        if highlights:
            for (t0, t1, c) in highlights:
                ax.axvspan(t0, t1, alpha=0.15, color=c)

    plt.tight_layout()
    plt.show()


def plot_comparison(t, sols, names, colors):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=DARK_BG)
    fig.suptitle('COMPARACIÓN DE CASOS', fontsize=15, fontweight='bold', color=TXT_CLR)

    pairs = [(0, 'u (m/s)'), (6, 'φ (rad)'), (7, 'θ (rad)'), (1, 'v (m/s)')]
    for idx, (ax, (si, lbl)) in enumerate(zip(axes.flat, pairs)):
        style_ax(ax, 'Tiempo (s)', lbl)
        for sol, name, col in zip(sols, names, colors):
            ax.plot(t, sol[:, si], color=col, linewidth=1.5, label=name)
        ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TXT_CLR)

    plt.tight_layout()
    plt.show()


# PSO - OPTIMIZACIÓN PARA VUELO TRIMADO

def cost_function_trim(params, Va_target=78.0, psi_target=np.radians(45)):
    
    alpha_val, delta_e, throttle = params

    u_val = Va_target * np.cos(alpha_val)
    w_val = Va_target * np.sin(alpha_val)
    theta_val = alpha_val  # gamma = theta - alpha = 0

    X = np.array([u_val, 0.0, w_val, 0.0, 0.0, 0.0, 0.0, theta_val, psi_target])
    U = np.array([0.0, delta_e, 0.0, throttle, throttle])

    try:
        xd = xdot(X, U)
    except Exception:
        return 1e10

    J = xd[0]**2 + xd[2]**2 + xd[4]**2

    return J


def run_pso(cost_func, lb, ub, n_particles=40, n_iter=100, w_inertia=0.7,
            c1=1.5, c2=1.5, seed=42):
    
    np.random.seed(seed)
    n_dims = len(lb)

    particles = np.random.uniform(lb, ub, (n_particles, n_dims))
    velocities = np.random.uniform(-1, 1, (n_particles, n_dims)) * (ub - lb) * 0.1

    pbest = particles.copy()
    pbest_cost = np.array([cost_func(p) for p in particles])

    gbest_idx = np.argmin(pbest_cost)
    gbest = pbest[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    history = []

    for it in range(n_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w_inertia * velocities[i]
                             + c1 * r1 * (pbest[i] - particles[i])
                             + c2 * r2 * (gbest - particles[i]))

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], lb, ub)

            cost = cost_func(particles[i])
            if cost < pbest_cost[i]:
                pbest[i] = particles[i].copy()
                pbest_cost[i] = cost
                if cost < gbest_cost:
                    gbest = particles[i].copy()
                    gbest_cost = cost

        history.append(gbest_cost)
        if (it + 1) % 20 == 0:
            print(f"   Iteración {it+1}/{n_iter}: Costo = {gbest_cost:.2e}")

    return gbest, gbest_cost, history


def plot_pso_results(history, t_verify, sol_verify, gbest, Va_target=78.0):
    
    alpha_opt, delta_e_opt, T_opt = gbest

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor=DARK_BG)
    style_ax(ax, 'Iteración', 'Costo J (log)', 'CONVERGENCIA PSO')
    ax.plot(range(1, len(history)+1), history, color='#E6C280', linewidth=2, marker='o', markersize=3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

    plot_case(t_verify, sol_verify,
              f'VUELO TRIMADO PSO — Va={Va_target} m/s, ψ=45°',
              color='#45A29E')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=DARK_BG)
    fig.suptitle('VERIFICACIÓN DE EQUILIBRIO — PSO', fontsize=14, fontweight='bold', color=TXT_CLR)

    Va_traj = np.sqrt(sol_verify[:, 0]**2 + sol_verify[:, 2]**2)
    ax = axes[0]
    style_ax(ax, 'Tiempo (s)', 'Va (m/s)', 'Velocidad Aerodinámica')
    ax.plot(t_verify, Va_traj, color='#45A29E', linewidth=2)
    ax.axhline(y=Va_target, color='#E74C3C', linestyle='--', linewidth=1.5, label=f'Va_obj = {Va_target}')
    ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TXT_CLR)

    ax = axes[1]
    style_ax(ax, 'Tiempo (s)', 'θ (°)', 'Ángulo de Cabeceo')
    ax.plot(t_verify, np.degrees(sol_verify[:, 7]), color='#27AE60', linewidth=2)
    ax.axhline(y=np.degrees(alpha_opt), color='#E74C3C', linestyle='--', linewidth=1.5)

    ax = axes[2]
    style_ax(ax, 'Tiempo (s)', 'φ (°)', 'Ángulo de Alabeo')
    ax.plot(t_verify, np.degrees(sol_verify[:, 6]), color='#8E44AD', linewidth=2)
    ax.axhline(y=0, color='#E74C3C', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.show()


def main():
    print("\n" + "=" * 70)
    print("  SIMULADOR RCAM — Task 3: Aircraft Model")
    print("=" * 70)

    # Condiciones iniciales
    x0 = np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    dt = 1.0               # Paso discreto de 1 segundo
    T_total = 180           # 3 minutos
    t_sim = np.arange(0, T_total + dt, dt)

    # CASO 1
    print("\n  [CASO 1] Vuelo Base (3 min, controles constantes)...")
    sol1 = simulate(x0, t_sim, controls_case1, dt=dt)
    plot_case(t_sim, sol1, 'CASO 1: VUELO BASE (180 s)', color='#3498DB')

    # CASO 2
    print("  [CASO 2] Deflexión Alerones +5° (t=30-32s)...")
    sol2 = simulate(x0, t_sim, controls_case2, dt=dt)
    plot_case(t_sim, sol2, 'CASO 2: DEFLEXIÓN ALERONES +5° (t=30-32s)',
              color='#27AE60', highlights=[(30, 32, '#E74C3C')])

    # CASO 3
    print("  [CASO 3] Falla de Motor (throttle1->0 en t>=30s)...")
    sol3 = simulate(x0, t_sim, controls_case3, dt=dt)
    plot_case(t_sim, sol3, 'CASO 3: FALLA DE MOTOR (t≥30s)',
              color='#E74C3C', highlights=[(30, 180, '#F39C12')])

    # Comparación
    print("  Generando gráfica comparativa...")
    plot_comparison(t_sim,
                    [sol1, sol2, sol3],
                    ['Base', 'Alerones', 'Motor'],
                    ['#3498DB', '#27AE60', '#E74C3C'])

    # CASO 4
    print("\n" + "=" * 70)
    print("  [CASO 4] PSO — Trim a 78 m/s, rumbo NE (ψ=45°)")
    print("=" * 70)

    Va_target = 78.0
    psi_target = np.radians(45.0)

    # Límites: [alpha, delta_e, throttle]
    lb = np.array([np.radians(-5), np.radians(-25), 0.01])
    ub = np.array([np.radians(15), np.radians(10),  0.20])

    print(f"\n   Partículas: 40 | Iteraciones: 100")
    print(f"   Variables: [α, δe, T] (3 dimensiones)")
    print(f"   Va objetivo: {Va_target} m/s | ψ objetivo: 45°\n")

    gbest, gbest_cost, history = run_pso(
        lambda p: cost_function_trim(p, Va_target, psi_target),
        lb, ub, n_particles=40, n_iter=100
    )

    alpha_opt, delta_e_opt, T_opt = gbest
    u_opt = Va_target * np.cos(alpha_opt)
    w_opt = Va_target * np.sin(alpha_opt)
    theta_opt = alpha_opt

    print(f"\n   ✓ PSO COMPLETADO — Costo final: {gbest_cost:.2e}")
    print(f"\n   ESTADO TRIM ENCONTRADO:")
    print(f"     u     = {u_opt:.4f} m/s")
    print(f"     v     = 0.0000 m/s")
    print(f"     w     = {w_opt:.4f} m/s")
    print(f"     p,q,r = 0.0 rad/s")
    print(f"     φ     = 0.0°")
    print(f"     θ     = {np.degrees(theta_opt):.4f}°")
    print(f"     ψ     = 45.0°")
    print(f"\n   CONTROLES TRIM:")
    print(f"     δa = 0.0°")
    print(f"     δe = {np.degrees(delta_e_opt):.4f}°")
    print(f"     δr = 0.0°")
    print(f"     T1 = T2 = {T_opt:.6f} ({T_opt*100:.2f}%)")
    print(f"\n   Va resultante: {Va_target:.4f} m/s (exacto por construcción)")

    # Verificación temporal del trim
    print("\n  Verificando estabilidad del trim (60 s)...")
    X_trim = np.array([u_opt, 0.0, w_opt, 0.0, 0.0, 0.0, 0.0, theta_opt, psi_target])
    U_trim = np.array([0.0, delta_e_opt, 0.0, T_opt, T_opt])

    t_verify = np.arange(0, 61, dt)
    sol_trim = simulate(X_trim, t_verify, lambda t: U_trim, dt=dt)

    # Estabilidad en últimos 20s
    stable = sol_trim[-20:, :]
    print(f"\n   ESTABILIDAD (últimos 20 s):")
    print(f"     u: {stable[:, 0].mean():.4f} ± {stable[:, 0].std():.6f} m/s")
    print(f"     w: {stable[:, 2].mean():.4f} ± {stable[:, 2].std():.6f} m/s")
    print(f"     θ: {np.degrees(stable[:, 7].mean()):.4f}° ± {np.degrees(stable[:, 7].std()):.6f}°")

    plot_pso_results(history, t_verify, sol_trim, gbest, Va_target)

    print("\n" + "=" * 70)
    print("  ✓ TAREA 3 COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
