import numpy as np
from math import *


def canal(mu, sigma, muestras, data):
    return data + np.random.normal(mu, sigma, muestras)


def map_detection(y, c, p, varianza):
    mensaje_decodificado_map = np.zeros(y.size, 'int64')

    mensaje_decodificado_map[y * ((c[1] - c[0]) / varianza) >= log(p[0] / p[1])] = 1
    mensaje_decodificado_map[y * ((c[1] - c[0]) / varianza) < log(p[0] / p[1])] = -1

    return mensaje_decodificado_map


def ml_detection(y, c, varianza):
    mensaje_decodificado_ml = np.zeros(y.size, 'int64')

    mensaje_decodificado_ml[y * ((c[1] - c[0]) / varianza) >= 0] = 1
    mensaje_decodificado_ml[y * ((c[1] - c[0]) / varianza) < 0] = -1

    return mensaje_decodificado_ml


def statistic_map_error(mu, var, mensaje_transmitido, c, p, muestras):
    error_estimado = np.zeros(var.size, 'float64')

    for i in range(var.size):
        y = canal(mu, var[i], muestras, mensaje_transmitido)
        mensaje_decodificado = map_detection(y, c, p, var[i])
        error_estimado[i] = np.sum(abs((mensaje_decodificado - mensaje_transmitido)) / 2) / muestras

    return error_estimado


def statistic_ml_error(mu, var, mensaje_transmitido, c, muestras):
    error_estimado = np.zeros(var.size, 'float64')

    for i in range(var.size):
        y = canal(mu, var[i], muestras, mensaje_transmitido)
        mensaje_decodificado = ml_detection(y, c, var[i])
        error_estimado[i] = np.sum(abs((mensaje_decodificado - mensaje_transmitido)) / 2) / muestras

    return error_estimado


def analytic_error(var, c, p0, p1):
    error_analitico = np.zeros(var.size, 'float64')
    n = log(p0 / p1)

    for i in range(var.size):
        tita = (var[i] / (c[1] - c[0])) * n + (c[1] + c[0]) / 2

        pe0 = 1 / 2 * erfc(((tita - c[0]) / np.sqrt(var[i])) / np.sqrt(2))
        pe1 = 1 / 2 * erfc(((c[1] - tita) / np.sqrt(var[i])) / np.sqrt(2))

        error_analitico[i] = (p0 * pe0) + (p1 * pe1)

    return error_analitico
