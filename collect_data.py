import os
import csv
import numpy as np
from experiments import Arquetipo, ParConDecoherencia

SEED = 42


def generar_dataset_arquetipo(n_shots=500, step=0.05):
    """
    Para cada alpha en [0, 1] mide la proporción de Ánima (|0>).

    La relación teórica exacta es P(0) = alpha², no lineal.
    El dataset generado permite comparar esa curva contra modelos de regresión.

    Args:
        n_shots: Número de mediciones por valor de alpha.
        step:    Incremento entre valores de alpha consecutivos.
    """
    np.random.seed(SEED)
    datos = []
    for alpha in np.arange(0, 1 + step, step):
        beta = np.sqrt(max(0.0, 1 - alpha**2))  # complemento normalizado
        arq  = Arquetipo(alpha, beta)
        resultados  = [arq.medir() for _ in range(n_shots)]
        prob_anima  = 1 - sum(resultados) / n_shots
        datos.append([round(alpha, 6), round(prob_anima, 6)])

    _guardar_csv(
        'datasets/arquetipo_prob.csv',
        ['alpha', 'prob_anima'],
        datos,
        'Dataset A guardado: arquetipo_prob.csv'
    )


def generar_dataset_sincronicidad(step=0.05, repeticiones=200, trials_por_rep=200):
    """
    Para cada gamma en [0, 1] estima la correlación media en base X.

    La relación teórica exacta es correlacion = 1 - gamma, perfectamente lineal.
    El modelo de regresión lineal debería obtener R² ≈ 1.0 sobre este dataset.

    Args:
        step:           Incremento entre valores de gamma consecutivos.
        repeticiones:   Número de instancias de ParConDecoherencia por gamma.
        trials_por_rep: Mediciones por instancia para estimar la correlación.
    """
    np.random.seed(SEED)
    datos = []
    for gamma in np.arange(0, 1 + step, step):
        correlaciones = []
        for _ in range(repeticiones):
            par = ParConDecoherencia()
            par.aplicar_represion(gamma)
            iguales = sum(
                1 for _ in range(trials_por_rep)
                if par.medir_base_X()[0] == par.medir_base_X()[1]
            )
            correlaciones.append(iguales / trials_por_rep)
        datos.append([round(gamma, 6), round(float(np.mean(correlaciones)), 6)])

    _guardar_csv(
        'datasets/sincronicidad_corr.csv',
        ['gamma', 'correlacion'],
        datos,
        'Dataset B guardado: sincronicidad_corr.csv'
    )


def _guardar_csv(path, encabezado, datos, mensaje):
    """Escribe un CSV con encabezado y muestra confirmación."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(encabezado)
        writer.writerows(datos)
    print(mensaje)


if __name__ == '__main__':
    generar_dataset_arquetipo()
    generar_dataset_sincronicidad()
