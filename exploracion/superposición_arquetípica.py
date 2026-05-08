import numpy as np
import matplotlib.pyplot as plt

class Arquetipo:
    """Qubit que modela un par de opuestos arquetípicos."""
    def __init__(self, alpha=1/np.sqrt(2), beta=1/np.sqrt(2)):
        # Estado normalizado alpha|0> + beta|1>
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        self.alpha = alpha / norm
        self.beta = beta / norm

    def medir(self):
        """Colapso: devuelve 0 (Ánima) o 1 (Ánimus) según probabilidades."""
        prob_0 = abs(self.alpha)**2
        return 0 if np.random.random() < prob_0 else 1

# Experimento: 1000 mediciones sobre el mismo estado.
arq = Arquetipo()
resultados = [arq.medir() for _ in range(1000)]
unos = sum(resultados)
ceros = len(resultados) - unos

print(f"Ánima (0): {ceros}, Ánimus (1): {unos}")
plt.bar(['Ánima (0)','Ánimus (1)'], [ceros, unos], color=['blue','red'])
plt.title("Distribución post-colapso: superposición inicial equilibrada")
plt.ylabel("Frecuencia")
plt.show()