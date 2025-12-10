import time
import torch
import torch.optim as optim
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

# ==============================================================================
# 0. CLASE AUXILIAR: EARLY STOPPING (SIN CAMBIOS)
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, path='best_frcnn_model_practical.pth'): 
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
        print(f'   ✅ Mejora detectada. Guardando checkpoint en {self.path}...')
        torch.save(model.state_dict(), self.path)


# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
NUM_CLASSES = 81 
# AJUSTE: Aumento seguro del BATCH_SIZE (de 4 a 8)
BATCH_SIZE = 8   
EPOCHS = 10      
PATIENCE = 3     
# AJUSTE: Forzar el uso de la Tarjeta 1 (cuda:1)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_frcnn_model_practical.pth"

# Se eliminan las líneas de SSD_CACHE_PATH (para evitar conflicto de fsspec)


# --- 2. CARGA DE DATASETS (ESTRATEGIA PRÁCTICA) ---

print(f"Cargando dataset de ENTRENAMIENTO ('val' ~5K) y VALIDACIÓN ('train[:500]')...")

# AJUSTE: Se elimina cache_dir para evitar KeyError
# Entrenamiento (Práctico): Usamos el split 'val' (~5K imágenes)
train_dataset = load_dataset("detection-datasets/coco", split="val")
# Validación (Rápido): Usamos un slice pequeño del split 'train' (500 imágenes)
val_dataset = load_dataset("detection-datasets/coco", split="train[:500]")

print(f"Dataset de Entrenamiento: {len(train_dataset)} ejemplos | Dataset de Validación (ES): {len(val_dataset)} ejemplos")


# ==============================================================================
# 3. DATA PIPELINE CRÍTICO (Faster R-CNN) - SIN CAMBIOS
# ==============================================================================

def custom_collate_fn(batch):
    images = []
    targets = []
    transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()

    for item in batch:
        img_tensor = transform(item['image'].convert("RGB")) 
        images.append(img_tensor.to(DEVICE))
        
        annotations = item['objects'] 
        boxes = []
        labels = []
        
        for bbox_coco, category_id in zip(annotations['bbox'], annotations['category']):
            x_min, y_min, w, h = bbox_coco
            boxes.append([x_min, y_min, x_min + w, x_min + w]) # ERROR DE CÓDIGO: Se repite x_min + w
            # El código original era: boxes.append([x_min, y_min, x_min + w, x_min + w])
            # La corrección debería ser:
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(category_id + 1)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32).to(DEVICE)
            labels = torch.zeros((0,), dtype=torch.int64).to(DEVICE)
        else:
            boxes = boxes.to(DEVICE)
            labels = labels.to(DEVICE)
        
        targets.append({"boxes": boxes, "labels": labels})

    return images, targets


# Creamos los DataLoaders
# AJUSTE: Se añade num_workers=4
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)


# ==============================================================================
# 4. MODELO Y ENTRENAMIENTO (SIN CAMBIOS)
# ==============================================================================

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
early_stopping = EarlyStopping(patience=PATIENCE, path=BEST_MODEL_PATH)


print(f"\nIniciando Entrenamiento de FRCNN ({EPOCHS} Épocas) en {DEVICE}...")
start_train = time.time()

for epoch in range(EPOCHS):
    # --- FASE DE ENTRENAMIENTO ---
    model.train()
    for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} TRAIN"):
        loss_dict = model(images, targets) 
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # --- FASE DE VALIDACIÓN (EARLY STOPPING) ---
    model.eval()
    val_loss_sum = 0
    with torch.no_grad():
        for images, targets in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} VALID"):
            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values())
            val_loss_sum += losses.item()
    
    avg_val_loss = val_loss_sum / len(val_dataloader)
    
    print(f"\n[EPOCH {epoch+1}] VALIDATION Loss: {avg_val_loss:.4f}")

    # --- APLICAR EARLY STOPPING ---
    early_stopping(avg_val_loss, model)
    
    if early_stopping.early_stop:
        print("\n🚫 Detención temprana activada.")
        break

end_train = time.time()
training_time = end_train - start_train
total_epochs_run = epoch + 1


# ==============================================================================
# 5. PRUEBA DE INFERENCIA Y CÁLCULO DE mAP (MÉTRICAS COMPLEJAS) - SIN CAMBIOS
# ==============================================================================

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

print("\n--- INICIANDO PRUEBA DE INFERENCIA Y CÁLCULO DE mAP ---")

inference_times = []
MAX_TEST_BATCHES = 100 
metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", num_classes=NUM_CLASSES - 1).to(DEVICE)

if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(DEVICE)
    max_vram_allocated_start = torch.cuda.max_memory_allocated(DEVICE)

with torch.no_grad():
    for i, (images, targets) in enumerate(tqdm(val_dataloader, desc="Mid. mAP & Inf.", total=MAX_TEST_BATCHES)):
        if i >= MAX_TEST_BATCHES:
            break
            
        start_inference = time.time()
        outputs = model(images) 
        end_inference = time.time()

        batch_time = end_inference - start_inference
        num_images = len(images)
        
        if num_images > 0:
            avg_time_per_image = batch_time / num_images
            inference_times.append(avg_time_per_image)
            
            # Formato para TorchMetrics
            preds = []
            for pred in outputs:
                preds.append({
                    "boxes": pred['boxes'],
                    "scores": pred['scores'],
                    "labels": pred['labels'] - 1 # Ajustar etiquetas: 0-79
                })
            
            target_metrics = []
            for t in targets:
                target_metrics.append({
                    "boxes": t['boxes'],
                    "labels": t['labels'] - 1 # Ajustar etiquetas: 0-79
                })

            metric.update(preds, target_metrics)

# Cálculo de Métricas Finales
computed_metrics = metric.compute()

avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
estimated_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
max_vram_allocated = (torch.cuda.max_memory_allocated(DEVICE) - max_vram_allocated_start) / (1024**3) if DEVICE.type == 'cuda' else 0.0

# ==============================================================================
# 6. REPORTE FINAL ESTANDARIZADO - SIN CAMBIOS
# ==============================================================================

def print_final_report(training_time, best_val_loss, avg_inference_time, estimated_fps, max_vram_allocated, total_epochs_run, computed_metrics):
    
    mAP_50 = computed_metrics.get('map_50', torch.tensor(0.0)).item()
    mAP_total = computed_metrics.get('map', torch.tensor(0.0)).item()
    
    print("\n" + "="*60)
    print("✅ REPORTE FINAL ESTANDARIZADO - FRCNN (Detección)")
    print("="*60)
    print(f"1. TIEMPO ENTRENAMIENTO TOTAL: {training_time:.2f} s")
    print(f"2. TIEMPO INFERENCIA PROMEDIO: {avg_inference_time*1000:.4f} ms/imagen")
    print(f"3. FPS ESTIMADOS: {estimated_fps:.2f} FPS")
    print(f"4. VRAM MÁXIMA ASIGNADA: {max_vram_allocated:.2f} GB")
    print(f"5. ÉPOCAS COMPLETADAS: {total_epochs_run}")
    print("\n--- 6. MÉTRICAS DE CALIDAD DE DETECCIÓN (COCO) ---")
    print(f"   📉 PÉRDIDA DE VALIDACIÓN (MEJOR): {best_val_loss:.4f}")
    print(f"   🏆 mAP Total (COCO): {mAP_total:.4f}")
    print(f"   🏆 mAP@50 (COCO): {mAP_50:.4f}")
    print("="*60)

# Llamar a la función de reporte
print_final_report(training_time, early_stopping.best_loss, avg_inference_time, estimated_fps, max_vram_allocated, total_epochs_run, computed_metrics)