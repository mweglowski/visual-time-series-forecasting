from models import VisualAE, EncoderPaper, DecoderPaper
import matplotlib.pyplot as plt
from data import load_x_y
import numpy as np
import torch

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(device):
    latent_dim = 512
    encoder = EncoderPaper(latent_dim)
    decoder = DecoderPaper(latent_dim)
    model = VisualAE(encoder, decoder)
    state_dict = torch.load('weights/weights_exp_20.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def plot_results(x_batch, y_batch, output_batch):
    output_batch = output_batch.detach().cpu().numpy()
    
    num_samples = x_batch.shape[0]
    print(f'Processing visualization for batch size: {num_samples}')

    _, axes = plt.subplots(num_samples, 3, figsize=(6, 2 * num_samples))
    
    for i in range(num_samples):
        x = np.squeeze(x_batch[i])
        y = np.squeeze(y_batch[i])
        out = np.squeeze(output_batch[i])
        axes[i, 0].imshow(x, cmap='gray')
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title('Sample History')
        axes[i, 1].imshow(out, cmap='gray')
        axes[i, 1].axis('off')
        if i == 0: axes[i, 1].set_title('VisualAE Prediction')
        axes[i, 2].imshow(y, cmap='gray')
        axes[i, 2].axis('off')
        if i == 0: axes[i, 2].set_title('Ground Truth')
        
    plt.tight_layout()
    plt.savefig('visualizations/model_batch_predictions_test_20.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def get_samples():
    X, Y = load_x_y('financial_x_y')
    n_samples = 5
    random_indices = np.random.randint(len(X) - 150, len(X) - 1, n_samples) # test
    # random_indices = np.random.randint(1000, 5000, n_samples) # train
    x_batch = X[random_indices]
    y_batch = Y[random_indices]
    return x_batch, y_batch

def infer(model, x_batch, device):
    x_tensor = torch.tensor(x_batch, device=device).unsqueeze(1)
    with torch.no_grad():
        output = model(x_tensor)
    return output

def main():
    device = get_device()
    x_batch, y_batch = get_samples()
    model = get_model(device)
    output_batch = infer(model, x_batch, device)
    plot_results(x_batch, y_batch, output_batch)

if __name__ == '__main__':
    main()