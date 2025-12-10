import time
import torch
import torch.optim as optim
import os
import torch.multiprocessing as mp 
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

# ==============================================================================
# 0. CLASE AUXILIAR: EARLY STOPPING (DEBE ESTAR FUERA DEL MAIN)
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
# 1. FUNCIÓN DE COLLATE (DEBE ESTAR FUERA DEL MAIN)
# ==============================================================================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

# ==============================================================================
# 2. FUNCIÓN DE REPORTE FINAL (FUERA DEL MAIN)
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


# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
if __name__ == '__main__':
    
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 

    # 2. CONFIGURACIÓN
    NUM_CLASSES = 81 
    # MODIFICACIÓN 1: Aumentar BATCH_SIZE a 64 para usar la VRAM de 80 GB
    BATCH_SIZE = 64   
    EPOCHS = 10      
    PATIENCE = 3     
    BEST_MODEL_PATH = "best_frcnn_model_practical.pth"
    
    print(f"--- Iniciando Benchmark en {DEVICE} ---")

    # --- 3. CARGA DE DATASETS ---
    # MODIFICACIÓN 2: Usar cache_dir local para leer desde ~/coco_data
    LOCAL_CACHE_DIR = "/data" 
    
    print(f"Cargando dataset de ENTRENAMIENTO (COMPLETO) y VALIDACIÓN (split='val') desde cache local...")
    
    # MODIFICACIÓN 3: Cargar el split="train" COMPLETO
    train_dataset = load_dataset("detection-datasets/coco", split="train", cache_dir=LOCAL_CACHE_DIR)
    
    # MODIFICACIÓN 4: Cargar el split="val" (oficial para validación)
    val_dataset = load_dataset("detection-datasets/coco", split="val", cache_dir=LOCAL_CACHE_DIR)

    print(f"Dataset de Entrenamiento: {len(train_dataset)} ejemplos | Dataset de Validación: {len(val_dataset)} ejemplos")

    # Creamos los DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)

    # --- 4. MODELO Y ENTRENAMIENTO ---
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_V1.transforms(min_size=800), weights_backbone=FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(min_size=800))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=PATIENCE, path=BEST_MODEL_PATH)

    print(f"\nIniciando Entrenamiento de FRCNN ({EPOCHS} Épocas)...")
    start_train = time.time()
    
    total_epochs_run = 0

    for epoch in range(EPOCHS):
        total_epochs_run = epoch + 1
        
        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} TRAIN"):
            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # --- FASE DE VALIDACIÓN ---
        model.train() 
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
    
    # --- 5. PRUEBA DE INFERENCIA Y CÁLCULO DE mAP ---
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    else:
        print("⚠️ Advertencia: No se encontró el checkpoint, usando el modelo actual.")

    model.eval()

    print("\n--- INICIANDO PRUEBA DE INFERENCIA Y CÁLCULO DE mAP ---")

    inference_times = []
    MAX_TEST_BATCHES = 100 
    
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(DEVICE)

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
                
                preds = []
                for pred in outputs:
                    preds.append({
                        "boxes": pred['boxes'],
                        "scores": pred['scores'],
                        "labels": pred['labels'] - 1 
                    })
                
                target_metrics = []
                for t in targets:
                    target_metrics.append({
                        "boxes": t['boxes'],
                        "labels": t['labels'] - 1 
                    })

                metric.update(preds, target_metrics)

    computed_metrics = metric.compute()

    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    estimated_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
    max_vram_allocated = (torch.cuda.max_memory_allocated(DEVICE) - max_vram_allocated_start) / (1024**3) if DEVICE.type == 'cuda' else 0.0

    print_final_report(training_time, early_stopping.best_loss, avg_inference_time, estimated_fps, max_vram_allocated, total_epochs_run, computed_metrics)