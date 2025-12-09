import time 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import load_dataset
from sklearn.metrics import classification_report
import numpy as np

# ==============================================================================
# 1. CLASE AUXILIAR: EARLY STOPPING Y CHECKPOINT (.pth)
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_resnet50_cnn.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'   ⚠️ EarlyStopping: {self.counter}/{self.patience} sin mejora. Actual: {val_loss:.6f} | Mejor: {self.best_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Guarda el modelo cuando el loss de validación disminuye.'''
        print(f'   ✅ Mejora detectada ({val_loss:.6f}). Guardando checkpoint en {self.path}...')
        torch.save(model.state_dict(), self.path)

# ==============================================================================
# 2. CONFIGURACIÓN DEL BENCHMARK
# ==============================================================================
# MODIFICACIÓN: Aumento el BATCH_SIZE para aprovechar la gran VRAM del supercomputador
BATCH_SIZE = 64          
EPOCHS = 10              
MODEL_NAME = "ResNet-50"
BEST_MODEL_PATH = "best_resnet50_cnn.pth"

# Usamos la GPU (CUDA) del supercomputador
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

print(f"--- Iniciando Benchmark en {DEVICE} con {MODEL_NAME} ---")

# 3. CARGAR Y PREPROCESAR DATASET
# Usamos un pequeño subset de train y lo dividimos en Train/Test
dataset = load_dataset("timm/mini-imagenet", split="train[:10%]")
dataset = dataset.train_test_split(test_size=0.2) # 80% Train, 20% Test (para validación)
NUM_CLASSES = 100 

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_data(examples):
    examples["pixel_values"] = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset = dataset.with_transform(transform_data)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return pixel_values, labels

# MODIFICACIÓN: Se añade num_workers=4 para carga de datos paralela
train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)

# 4. DEFINIR MODELO, LOSS Y OPTIMIZADOR
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES) # Modificar la capa final
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping = EarlyStopping(patience=5, path=BEST_MODEL_PATH)

# ==============================================================================
# 5. BUCLE DE ENTRENAMIENTO CON EARLY STOPPING
# ==============================================================================
print("Iniciando Entrenamiento...")
start_train = time.time()

for epoch in range(EPOCHS):
    # --- FASE DE ENTRENAMIENTO ---
    model.train()
    train_losses = []
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # --- FASE DE VALIDACIÓN (Evalúa el éxito del entrenamiento) ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses)
    avg_train_loss = np.mean(train_losses)
    
    print(f'Época {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f}')
    
    # --- APLICAR EARLY STOPPING Y GUARDADO .pth ---
    early_stopping(avg_val_loss, model)
    
    if early_stopping.early_stop:
        break

end_train = time.time()
training_time = end_train - start_train

# 6. CARGAR EL MEJOR MODELO PARA INFERENCIA FINAL
model.load_state_dict(torch.load(BEST_MODEL_PATH))

# ==============================================================================
# 7. INFERENCIA Y REPORTE DE MÉTRICAS (Hardware y Modelo)
# ==============================================================================
print("\n" + "="*50)
print(f"✅ INFERENCIA FINAL Y REPORTE DE RENDIMIENTO ({MODEL_NAME})")
print("="*50)

model.eval()
all_preds = []
all_labels = []
inference_times = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        start_inf = time.time()
        outputs = model(images)
        end_inf = time.time()
        
        inference_times.append((end_inf - start_inf) / len(images))
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_inference = np.mean(inference_times) * 1000 # a milisegundos

print(f"1. TIEMPO ENTRENAMIENTO TOTAL: {training_time:.2f} s")
print(f"2. TIEMPO INFERENCIA PROMEDIO: {avg_inference:.4f} ms/imagen")
print(f"3. FPS ESTIMADOS: {1000/avg_inference:.2f} FPS")

if torch.cuda.is_available():
    print(f"4. VRAM MÁXIMA ASIGNADA: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

print("\n--- 5. MÉTRICAS DE CLASIFICACIÓN (Modelo) ---")
print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

# 8. EXPORTAR A ONNX (Requerimiento de Interoperabilidad)
try:
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    onnx_path = "resnet50_despliegue.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'], opset_version=12)
    print(f"\n✅ Archivo de Interoperabilidad: {onnx_path} creado.")
except Exception as e:
    print(f"\n⚠️ ERROR al exportar a ONNX: {e}")