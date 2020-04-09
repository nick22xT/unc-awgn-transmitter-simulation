import matplotlib.pyplot as plt

from bpsk_modules import *


def imprimir_histograma(nombre, datos):
    plt.title(nombre)
    plt.hist(datos, 8)
    plt.grid()
    plt.show()
    plt.clf()


muestras = 100300
h = np.array([1, 2])  # Simbolos de entrada al transmisor
c = np.array([-1, 1])  # Valores de la señal antipodal para los simbolos de entrada
p = np.array([0.3, 0.7])  # Probabilidad de ocurrencia de los mensajes

# Generador de mensajes
mensaje = np.random.choice(h, size=muestras, p=p)

mensaje_codificado = np.zeros(mensaje.size, 'int64')
mensaje_codificado[mensaje == 1] = 0
mensaje_codificado[mensaje == 2] = 1

print("# Lista de mensajes a transmitir: ", mensaje)
print("")
print("# Lista de mensajes codificados: ", mensaje_codificado)

imprimir_histograma("Histograma: Mensajes", mensaje)

# Modulador

mensaje_modulado = np.zeros(mensaje.size, 'int64')
mensaje_modulado[mensaje_codificado == 0] = c[0]
mensaje_modulado[mensaje_codificado == 1] = c[1]

print("")
print("# Mensajes modulados: ", mensaje_modulado)

imprimir_histograma("Histograma: Mensajes Modulados", mensaje_modulado)

# Modelado del canal

mu, sigma = 0, 1
y = canal(mu, sigma, muestras, mensaje_modulado)

print("")
print("# Mensajes modulados a la salida del canal: ", y)
imprimir_histograma("Histograma: Mensajes a la salida del canal", y)

# Decodificador
varianza = pow(sigma, 2)

# MAP
mensaje_decodificado_map = map_detection(y, c, p, varianza)

# ML
mensaje_decodificado_ml = ml_detection(y, c, varianza)

print("")
print("# Mensaje decodificado con MAP: ", mensaje_decodificado_map)
print("")
print("# Mensaje decodificado con ML: ", mensaje_decodificado_ml)

imprimir_histograma("Histograma: Mensajes Decodificados con MAP", mensaje_decodificado_map)
imprimir_histograma("Histograma: Mensajes Decodificados con ML", mensaje_decodificado_ml)

# Deteccion de errores
varianzas = np.arange(0.1, 1, 0.1)

error_estimado_map = statistical_map_error(mu, varianzas, mensaje_modulado, c, p, muestras)
error_estimado_ml = statistical_ml_error(mu, varianzas, mensaje_modulado, c, muestras)

# Error Teorico
error_analitico_map = analytic_error(varianzas, c, p[0], p[1])
error_analitico_ml = analytic_error(varianzas, c, 0.5, 0.5)

plt.plot(varianzas, error_estimado_map)
plt.plot(varianzas, error_analitico_map)
plt.title("Relacion: Error analitico - estimado (MAP)")
plt.grid()
plt.semilogy()
plt.show()

plt.plot(varianzas, error_estimado_ml)
plt.plot(varianzas, error_analitico_ml)
plt.title("Relacion: Error analitico - estimado (ML)")
plt.grid()
plt.semilogy()
plt.show()

