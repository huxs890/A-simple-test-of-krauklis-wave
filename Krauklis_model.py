from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root

# ============================================================
# 1. Input parameters
# ============================================================
@dataclass(frozen=True)
class KrauklisParameters:
    # Fluid
    rho_f: float = 1000.0       # kg/m^3
    eta: float = 1.0e-3      # Pa*s
    Kf: float = 2.2e9           # Pa
    # Fracture
    w0: float = 1.0e-3       # m, FULL aperture
    L: float = 10.0          # m
    # Isotropic Solid
    rho_s: float = 2600.0    # kg/m^3
    cp_s: float = 5200.0   # m/s
    cs_s: float = 3000.0   # m/s
    def __post_init__(self):
        names = ("rho_f", "eta", "Kf", "w0","L", "rho_s", "cp_s", "cs_s")
        for name in names:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:raise ValueError(f"{name} must be positive and finite")
        if self.cp_s <= np.sqrt(4.0 / 3.0) * self.cs_s:
            raise ValueError(
                "A stable isotropic host requires "
                "cp_s > sqrt(4/3) * cs_s"
            )
    @property
    def mu(self) -> float:
        """Solid shear modulus [Pa]"""
        return self.rho_s * self.cs_s**2
    @property
    def G_star(self) -> float:
        """Equivalent quasistatic wall modulus [Pa]"""
        r2 = (self.cs_s / self.cp_s)**2
        return 2.0 * self.mu * (1.0 - r2)
# ============================================================
# 2. Output container
# ============================================================
@dataclass
class KrauklisSpectrum:
    frequency_hz: np.ndarray
    k: np.ndarray
    relative_error: np.ndarray
    H_end: np.ndarray
    dynamic_wall: bool
    @property
    def phase_velocity(self) -> np.ndarray:
        """Phase velocity [m/s]"""
        omega = 2.0 * np.pi * self.frequency_hz
        return omega / self.k.real
    @property
    def wavelength(self) -> np.ndarray:
        """Wavelength [m]."""
        return 2.0 * np.pi / self.k.real
    @property
    def attenuation(self) -> np.ndarray:
        """Spatial attenuation [Np/m]"""
        return self.k.imag
    @property
    def amplification(self) -> np.ndarray:
        """Finite-fracture pressure amplification"""
        return np.abs(self.H_end)
    @property
    def phase_deg(self) -> np.ndarray:
        """Pressure transfer phase [degree]"""
        return np.angle(self.H_end, deg=True)
    
# ============================================================
# 3. Krauklis wave model
# ============================================================
class KrauklisModel:
    def __init__(self, params: KrauklisParameters):
        self.p = params
    # --------------------------------------------------------
    # 3.1 Fluid dynamic correction
    # --------------------------------------------------------
    def phi(self, omega: float) -> complex:
        """
        Dynamic correction Phi(omega).
        Time convention: exp[i(k*x - omega*t)]
        omega: angular frequency [rad/s]
        """
        p = self.p
        xi = 0.5 * p.w0 * np.sqrt(1j * p.rho_f * omega / p.eta)
        # Small-xi Taylor expansion
        if abs(xi) < 1.0e-3:
            return (
                -xi**2 / 3.0
                - 2.0 * xi**4 / 15.0
                - 17.0 * xi**6 / 315.0
            )
        return 1.0 - np.tan(xi) / xi
    # --------------------------------------------------------
    # 3.2 Fracture-wall compliance
    # --------------------------------------------------------
    def wall_compliance(
        self,
        k: complex,
        omega: float,
        dynamic: bool = True
    ) -> complex:
        """
        Dynamic or quasistatic wall compliance.
        k: complex wavenumber [1/m]
        omega: angular frequency [rad/s]
        """
        p = self.p
        if not dynamic:
            return 2.0 / (p.G_star * k)
        v = omega / (p.cs_s * k)
        r2 = (p.cs_s / p.cp_s)**2
        P = np.sqrt(1.0 - r2 * v**2 + 0j)
        S = np.sqrt(1.0 - v**2 + 0j)
        # Numerically reduced Rayleigh denominator
        R = 4.0 - v**2 - 4.0 * (1.0 + r2 - r2 * v**2) / (1.0 + P * S)
        return 2.0 * P / (p.mu * k * R)
    # --------------------------------------------------------
    # 3.3 Dispersion equation
    # --------------------------------------------------------
    def dispersion(
        self,
        k: complex,
        omega: float,
        dynamic: bool = True
    ) -> complex:
        """
        D(k, omega) = Phi(w)*k^2 - rho_f*w^2/Kf - rho_f*w^2*Cw(k,w)/w0
        D(k, omega) = 0
        Returns the complex dispersion residual.
        """
        p = self.p
        cw = self.wall_compliance(k, omega, dynamic=dynamic)
        return (
            self.phi(omega) * k**2
            - p.rho_f * omega**2 / p.Kf
            - p.rho_f * omega**2 * cw / p.w0
        )
    # --------------------------------------------------------
    # 3.4 Initial guess: quasistatic cubic
    # --------------------------------------------------------
    def initial_guess(self, omega: float) -> complex:
        """
        Solve the quasistatic-wall cubic equation.
        Select the outgoing, attenuating root.
        """
        p = self.p
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError(
                "omega must be positive [rad/s]"
            )
        coeff = [self.phi(omega),0.0,-p.rho_f * omega**2 / p.Kf, -2.0 * p.rho_f * omega**2 / (p.G_star * p.w0)]
        roots = np.roots(np.asarray(coeff, dtype=complex))
        good = [
            z for z in roots
            if (z.real > 0.0 and z.imag >= -1.0e-10 * max(1.0, abs(z)))
        ]

        if len(good) != 1:
            raise RuntimeError(
                f"Cannot select physical root: {roots}"
            )
        return complex(good[0])
    # --------------------------------------------------------
    # 3.5 Solve at a single frequency
    # --------------------------------------------------------
    def solve_one(
        self,
        frequency_hz: float,
        dynamic: bool = True,
        k_guess: complex | None = None
    ) -> tuple[complex, float]:
        """
        Input:
            frequency_hz: frequency [Hz]
            dynamic: use dynamic elastic walls
            k_guess: optional initial complex wavenumber
        Returns:
            k: complex wavenumber [1/m]
            error: normalized dispersion residual
        """
        if (not np.isfinite(frequency_hz) or frequency_hz <= 0):
            raise ValueError(
                "Frequency must be positive and finite"
            )
        p = self.p
        omega = 2.0 * np.pi * frequency_hz
        k_qs = self.initial_guess(omega)
        kref = abs(k_qs)
        scale = (
            abs(self.phi(omega)) * kref**2
            + p.rho_f * omega**2 / p.Kf
            + 2.0 * p.rho_f * omega**2
            / (p.G_star * p.w0 * kref)
        )
        # The quasistatic cubic is already solved
        if not dynamic:
            error = abs(self.dispersion(k_qs, omega, dynamic=False)) / scale
            return k_qs, float(error)
    
        def residual(x):
            k = kref * complex(x[0], x[1])
            if abs(k) < 1.0e-14 * kref:
                return [1.0e8, 1.0e8]
            value = self.dispersion(k, omega, dynamic=True) / scale
            return [float(value.real),float(value.imag)]
        # Try continuation first, then cubic root
        guesses = [k_qs] if k_guess is None else [k_guess, k_qs]
        last_message = ""
        for guess in guesses:
            x0 = [complex(guess).real / kref, complex(guess).imag / kref]
            sol = root(residual,x0,method="hybr",options={"xtol": 1.0e-11,"maxfev": 300})
            k = kref * complex(sol.x[0], sol.x[1])
            error = abs(self.dispersion(k, omega, dynamic=True)) / scale
            valid = (
                np.isfinite(error)
                and error <= 1.0e-8
                and k.real > 0.0
                and k.imag >=
                -1.0e-10 * max(1.0, abs(k))
            )
            if valid:
                return k, float(error)
            last_message = (
                f"k={k}, error={error:.3e}, "
                f"status={sol.message}"
            )
        raise RuntimeError(
            f"Root failure at {frequency_hz:g} Hz: "
            f"{last_message}"
        )
    # --------------------------------------------------------
    # 3.6 Finite-fracture pressure transfer
    # --------------------------------------------------------
    def pressure_transfer(
        self,
        k: np.ndarray | complex,
        x: float | None = None
    ) -> np.ndarray:
        """
        Pressure transfer H(x) = p(x)/p(0).
        Boundary conditions:
            p(0) = prescribed pressure
            dp/dx(L) = 0
        Analytic expression: H(x) = cos[k*(L-x)] / cos(k*L)
        Uses a stable exponential representation.
        """
        p = self.p
        x = p.L if x is None else float(x)
        if not 0.0 <= x <= p.L:
            raise ValueError(
                f"x must lie in [0, {p.L}]"
            )
        k = np.asarray(k, dtype=complex)
        e2L = np.exp(2j * k * p.L)
        return np.exp(1j*k*x) * (1.0+np.exp(2j*k*(p.L-x)))/(1.0+e2L)
    # --------------------------------------------------------
    # 3.7 Frequency sweep
    # --------------------------------------------------------
    def solve_spectrum(
        self,
        frequencies_hz: np.ndarray,
        dynamic: bool = True,
        continuation: bool = True
    ) -> KrauklisSpectrum:
        """
        Solve the dispersion relation over frequencies.
        Frequencies are internally sorted for continuation.
        The returned arrays retain the input ordering.
        """
        f = np.asarray(frequencies_hz, dtype=float)
        if (
            f.ndim != 1
            or f.size == 0
            or not np.all(np.isfinite(f))
            or np.any(f <= 0)
        ):
            raise ValueError(
                "Provide a nonempty 1D array "
                "of positive frequencies"
            )
        order = np.argsort(f)
        k = np.empty(f.size, dtype=complex)
        error = np.empty(f.size, dtype=float)
        previous_k = None
        for index in order:
            k[index], error[index] = self.solve_one(
                float(f[index]),
                dynamic=dynamic,
                k_guess=(previous_k if continuation else None)
            )
            previous_k = (
                k[index] if continuation else None
            )
        H_end = self.pressure_transfer(
            k, x=self.p.L
        )
        return KrauklisSpectrum(
            frequency_hz=f,
            k=k,
            relative_error=error,
            H_end=H_end,
            dynamic_wall=dynamic
        )