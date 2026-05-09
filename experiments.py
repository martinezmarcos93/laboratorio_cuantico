import numpy as np


class Arquetipo:
    """
    Qubit arquetípico que modela un par de opuestos (Ánima/Ánimus).

    El estado es alpha|0> + beta|1>, normalizado automáticamente.
    La medición colapsa el estado: 0 = Ánima, 1 = Ánimus.

    Args:
        alpha: Amplitud del polo Ánima (|0>).
        beta:  Amplitud del polo Ánimus (|1>).
        seed:  Semilla aleatoria opcional para reproducibilidad.

    Raises:
        ValueError: Si alpha y beta son ambos cero (vector de norma nula).
    """
    def __init__(self, alpha, beta, seed=None):
        if seed is not None:
            np.random.seed(seed)
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        # BUG FIX: norma cero produce NaN silencioso — validar explícitamente
        if norm == 0:
            raise ValueError(
                "Las amplitudes alpha y beta no pueden ser ambas cero: "
                "el estado cuántico requiere una norma distinta de cero."
            )
        self.alpha = alpha / norm
        self.beta  = beta / norm

    def medir(self):
        """Colapso en base computacional. Retorna 0 (Ánima) o 1 (Ánimus)."""
        return 0 if np.random.random() < abs(self.alpha)**2 else 1


class ParConDecoherencia:
    """
    Dos qubits en estado de Bell |Φ+⟩, con canal de desfase aplicable al qubit 1.

    Los proyectores de medición en base X se precalculan en __init__ para
    evitar recomputarlos en cada llamada a medir_base_X().
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        self.rho = np.outer(phi_plus, phi_plus.conj())

        H  = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        I2 = np.eye(2)
        # Proyectores para qubit 1 en base X
        self.P0_1 = np.kron(np.outer(H[:, 0], H[:, 0].conj()), I2)
        self.P1_1 = np.kron(np.outer(H[:, 1], H[:, 1].conj()), I2)
        # Proyectores para qubit 2 en base X
        self.P0_2 = np.kron(I2, np.outer(H[:, 0], H[:, 0].conj()))
        self.P1_2 = np.kron(I2, np.outer(H[:, 1], H[:, 1].conj()))

    def aplicar_represion(self, gamma):
        """
        Aplica un canal de desfase (dephasing) al primer qubit con intensidad gamma.

        Operadores de Kraus:
            K0 = sqrt(1 - gamma) * I  →  evolución sin error
            K1 = sqrt(gamma) * Z⊗I   →  error de fase sobre qubit 1

        IMPORTANTE: gamma es la probabilidad de error por cada llamada a este método,
        no una tasa temporal acumulada. gamma=0 preserva el entrelazamiento intacto;
        gamma=1 aplica Z⊗I con certeza, convirtiendo |Φ+⟩ en |Φ-⟩ (anticorrelación
        perfecta en base X, no estado mixto sin correlación).

        Args:
            gamma: Probabilidad de desfase en [0, 1].

        Raises:
            ValueError: Si gamma está fuera del intervalo [0, 1]. Un valor fuera de
                        rango produciría sqrt de un número negativo (nan silencioso).
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(
                f"gamma debe estar en [0, 1]; se recibió {gamma}. "
                "Un valor fuera de rango produce sqrt de número negativo (nan silencioso)."
            )
        Z  = np.array([[1, 0], [0, -1]])
        I2 = np.eye(2)
        I4 = np.eye(4)
        K0 = np.sqrt(1 - gamma) * I4
        K1 = np.sqrt(gamma)     * np.kron(Z, I2)
        self.rho = K0 @ self.rho @ K0.conj().T + K1 @ self.rho @ K1.conj().T

    def medir_base_X(self):
        """
        Medición secuencial de ambos qubits en la base X (|+>, |->).

        El orden es qubit 1 primero, luego qubit 2 sobre el estado post-colapso.
        Estadísticamente equivalente a medir qubit 2 primero: el orden no afecta
        las probabilidades marginales ni las correlaciones observadas.

        Retorna:
            (x1, x2): resultados de medición, cada uno 0 (|+>) o 1 (|->).
        """
        prob0_q1 = np.trace(self.P0_1 @ self.rho).real
        if np.random.random() < prob0_q1:
            x1          = 0
            estado_post = self.P0_1 @ self.rho @ self.P0_1.conj().T
        else:
            x1          = 1
            estado_post = self.P1_1 @ self.rho @ self.P1_1.conj().T
        estado_post = estado_post / np.trace(estado_post)

        prob0_q2 = np.trace(self.P0_2 @ estado_post).real
        x2 = 0 if np.random.random() < prob0_q2 else 1
        return x1, x2
