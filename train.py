import matplotlib.pyplot as plt
import torch
import time
import os

from data import build_dataloaders, load_x_y, convert_x_y_to_tensors
from utils import extract_1d_prices, get_exp_num, log_experiment
from metrics import jsd_loss, mase, iou

def configure_optimization(model, init_lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=init_lr, 
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5,
    )
    return optimizer, scheduler

def train_visual_model(model, train_loader, val_loader, 
                       exp_num, epochs=50, device='cuda'):
    model = model.to(device)
    optimizer, scheduler = configure_optimization(model, init_lr=1e-3)

    best_val_loss = float('inf')
    early_stop_counter = 0

    history = {'train_loss': [],
               'val_loss': [],
               'val_mase': [],
               'val_iou': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs} started...')
        
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = jsd_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mase = 0.0
        val_iou_score = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                loss = jsd_loss(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # Calculate 1D Bounding Box IoU directly on the 2D visual outputs
                val_iou_score += iou(outputs, batch_y, device).item() * batch_x.size(0)
                
                # Cross-Metric Evaluation (Convert 2D Images to 1D continuous vectors)
                pred_prices = extract_1d_prices(outputs.squeeze(1))
                true_prices = extract_1d_prices(batch_y.squeeze(1))
                val_mase += mase(pred_prices, true_prices).item() * batch_x.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mase = val_mase / len(val_loader.dataset)
        avg_val_iou = val_iou_score / len(val_loader.dataset)

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mase'].append(avg_val_mase)
        history['val_iou'].append(avg_val_iou)

        print(f'Epoch [{epoch+1:02d}/{epochs}] '
              f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MASE: {avg_val_mase:.4f} | Val IoU: {avg_val_iou:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'weights/weights_exp_{exp_num}.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 15:
                print(f' Early stopping triggered at epoch {epoch+1}. Training terminated.')
                break

    return history

def create_train_history_charts(history, filename):
    os.makedirs('visualizations/training', exist_ok=True)
    # Expanded plot horizontally to accommodate the 3rd panel comfortably
    plt.figure(figsize=(18, 5))

    # 1. Loss Convergence (JSD)
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train JSD Loss')
    plt.plot(history['val_loss'], label='Val JSD Loss')
    plt.title('Loss Convergence History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Time-Series Error (MASE)
    plt.subplot(1, 3, 2)
    plt.plot(history['val_mase'], label='Val MASE', color='purple')
    plt.title('MASE Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('MASE Score')
    plt.legend()
    plt.grid(True)

    # 3. Spatial Localization Accuracy (1D IoU)
    plt.subplot(1, 3, 3)
    plt.plot(history['val_iou'], label='Val 1D IoU', color='teal')
    plt.title('IoU Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    chart_path = f'visualizations/training/{filename}.jpg'
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def main():
    description = 'Harmonic data '
    dataset = 'harmonic'

    batch_size = 128
    latent_dim = 512
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X, Y = load_x_y('financial_x_y')
    X_tensor, Y_tensor = convert_x_y_to_tensors(X, Y)
    train_loader, val_loader, test_loader = \
        build_dataloaders(X_tensor, Y_tensor, batch_size=batch_size)
    
    from models import VisualAE, EncoderPaper, DecoderPaper
    encoder = EncoderPaper(latent_dim)
    decoder = DecoderPaper(latent_dim)
    model = VisualAE(encoder, decoder)

    exp_file = 'experiments.md'
    exp_num = get_exp_num(exp_file)
    chart_filename = f'train_history_{exp_num}'

    start_time = time.perf_counter()
    history = train_visual_model(model, train_loader, val_loader, 
                                 exp_num, epochs, device)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f'Training took: {execution_time:.6f} seconds')
    
    completed_epochs = len(history['train_loss'])
    
    saved_chart_path = create_train_history_charts(history, chart_filename)

    params = {'batch_size': batch_size,
              'latent_dim': latent_dim,
              'epochs': epochs,
              'epochs_completed': completed_epochs,
              'device': device,
              'dataset': dataset}
    
    metrics = {
        'execution_time': execution_time,
        'best_train_jsd': min(history['train_loss']),
        'final_train_jsd': history['train_loss'][-1],
        'best_val_jsd': min(history['val_loss']),
        'final_val_jsd': history['val_loss'][-1],
        'best_val_mase': min(history['val_mase']),
        'final_val_mase': history['val_mase'][-1],
        'best_val_iou': max(history['val_iou']),
        'final_val_iou': history['val_iou'][-1]
    }
    
    log_experiment(filepath=exp_file,
                   params=params,
                   metrics=metrics,
                   model=model,
                   chart_path=saved_chart_path,
                   description=description)

if __name__ == '__main__':
    main()