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
    """
    def __init__(self, alpha, beta, seed=None):
        if seed is not None:
            np.random.seed(seed)
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        self.alpha = alpha / norm
        self.beta  = beta / norm

    def medir(self):
        """Colapso en base computacional. Retorna 0 (Ánima) o 1 (Ánimus)."""
        return 0 if np.random.random() < abs(self.alpha)**2 else 1


class ParConDecoherencia:
    """
    Par de qubits entrelazados en el estado de Bell |Φ+> = (|00> + |11>) / sqrt(2).

    Modela la sincronicidad junguiana: correlación perfecta entre un contenido
    interno y un evento externo. La decoherencia (represión) destruye esta
    correlación de forma gradual y proporcional.

    Args:
        seed: Semilla aleatoria opcional para reproducibilidad.
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        self.rho = np.outer(phi_plus, phi_plus.conj())

    def aplicar_represion(self, gamma):
        """
        Aplica un canal de desfase (dephasing) al primer qubit con intensidad gamma.

        Operadores de Kraus:
            K0 = sqrt(1 - gamma) * I  →  evolución sin error
            K1 = sqrt(gamma) * Z⊗I   →  error de fase sobre qubit 1

        IMPORTANTE: gamma es la probabilidad de error por cada llamada a este método,
        no una tasa temporal acumulada. gamma=0 preserva el entrelazamiento intacto;
        gamma=1 destruye completamente la coherencia entre los qubits (represión total).

        Args:
            gamma: Probabilidad de desfase en [0, 1].
        """
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
        Estadísticamente esto es equivalente a medir qubit 2 primero: el orden
        no afecta las probabilidades marginales ni las correlaciones observadas,
        pero sí determina cuál qubit colapsa el estado compartido.

        Retorna:
            (x1, x2): resultados de medición, cada uno 0 (|+>) o 1 (|->).
        """
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        # Proyectores del qubit 1 en base X, actuando sobre el espacio completo
        P0_1 = np.kron(np.outer(H[:, 0], H[:, 0].conj()), np.eye(2))
        P1_1 = np.kron(np.outer(H[:, 1], H[:, 1].conj()), np.eye(2))

        # Medir qubit 1
        prob0_q1 = np.trace(P0_1 @ self.rho).real
        if np.random.random() < prob0_q1:
            x1          = 0
            estado_post = P0_1 @ self.rho @ P0_1.conj().T
        else:
            x1          = 1
            estado_post = P1_1 @ self.rho @ P1_1.conj().T
        estado_post = estado_post / np.trace(estado_post)

        # Proyectores del qubit 2 en base X, actuando sobre el espacio completo
        P0_2 = np.kron(np.eye(2), np.outer(H[:, 0], H[:, 0].conj()))
        P1_2 = np.kron(np.eye(2), np.outer(H[:, 1], H[:, 1].conj()))

        # Medir qubit 2 sobre el estado colapsado por la medición anterior
        prob0_q2 = np.trace(P0_2 @ estado_post).real
        x2 = 0 if np.random.random() < prob0_q2 else 1

        return x1, x2
