from models import VisualAE, EncoderPlainConv, DecoderPlainConv
from data import build_dataloaders, load_x_y, convert_x_y_to_tensors
from metrics import jsd_loss, mase
from utils import extract_1d_prices
import torch

def configure_optimization(model, init_lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=init_lr, 
        weight_decay=weight_decay
    )
    # Reduce learning rate 10 times (by 0.1) if JSD loss stalled for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5,
    )
    return optimizer, scheduler

def train_visual_model(model, train_loader, val_loader, epochs=50, device='cuda'):
    model = model.to(device)
    optimizer, scheduler = configure_optimization(model, init_lr=1e-3)

    best_val_loss = float('inf')
    early_stop_counter = 0

    history = {'train_loss': [],
               'val_loss': [],
               'val_mase': []}

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
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                loss = jsd_loss(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # Cross-Metric Evaluation (Convert 2D Images to 1D continuous vectors)
                pred_prices = extract_1d_prices(outputs.squeeze(1))
                true_prices = extract_1d_prices(batch_y.squeeze(1))
                
                val_mase += mase(pred_prices, true_prices).item() * batch_x.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mase = val_mase / len(val_loader.dataset)

        # Step Scheduler using the Validation Loss scalar
        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mase'].append(avg_val_mase)

        print(f'Epoch [{epoch+1:02d}/{epochs}] '
              f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ')

        # Tracking & Early Stopping (15 Epoch Bounds)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'visual_ae_plain_conv.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 15:
                print(f' Early stopping triggered at epoch {epoch+1}. Training terminated.')
                break

    return history

def main():
    batch_size = 32
    latent_dim = 128

    X, Y = load_x_y('financial_x_y')
    X_tensor, Y_tensor = convert_x_y_to_tensors(X, Y)
    train_loader, val_loader, test_loader = \
        build_dataloaders(X_tensor, Y_tensor, batch_size=batch_size)
    
    encoder = EncoderPlainConv(latent_dim)
    decoder = DecoderPlainConv(latent_dim)
    model = VisualAE(encoder, decoder)
    train_visual_model(model, train_loader, val_loader)

if __name__ == '__main__':
    main()