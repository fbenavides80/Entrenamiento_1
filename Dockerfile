# Usa una imagen de NVIDIA que incluye PyTorch y soporte CUDA/cuDNN
# Usaremos una versión reciente (23.10) para aprovechar las GPUs del supercomputador
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el script de Python
COPY Bench_Inet_Clas_CNN.py /app/

# Instalar dependencias adicionales que su script necesita
# datasets y scikit-learn
RUN pip install --no-cache-dir datasets scikit-learn

# Comando por defecto para ejecutar su script
CMD ["python", "Bench_Inet_Clas_CNN.py"]