class ParSincronistico:
    """Dos arquetipos entrelazados (estado Phi+)."""
    def __init__(self):
        # Estado del sistema completo: a|00> + b|01> + c|10> + d|11>
        # Phi+: a = 1/sqrt(2), d = 1/sqrt(2), b=c=0
        self.estado = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

    def medir_ambos(self):
        """Medición conjunta: devuelve (interno, externo) correlacionados."""
        prob_00 = abs(self.estado[0])**2  # |00>
        prob_11 = abs(self.estado[3])**2  # |11>
        r = np.random.random()
        if r < prob_00:
            return 0, 0
        else:
            return 1, 1

# Experimento: 2000 mediciones del par entrelazado
par = ParSincronistico()
mediciones = [par.medir_ambos() for _ in range(2000)]
correlacion = sum(1 for i, e in mediciones if i == e) / len(mediciones)
print(f"Correlación (sincronicidad): {correlacion:.3f}")

# Visualización de los primeros 50 pares
internos, externos = zip(*mediciones[:50])
plt.scatter(range(50), internos, label='Interno (sueño)', marker='o')
plt.scatter(range(50), externos, label='Externo (evento)', marker='x')
plt.yticks([0,1], ['Ánima', 'Ánimus'])
plt.xlabel("Ensayo")
plt.ylabel("Resultado")
plt.legend()
plt.title("Sincronicidad: pares entrelazados")
plt.show()