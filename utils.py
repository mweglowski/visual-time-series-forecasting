from contextlib import contextmanager
import torch
import time
import os
import re

def extract_1d_prices(y_image_matrix):
    '''
    Converts 2D probability grid [B, H, W] to 1D price index tracking [B, W]
    Usually used when we have image (matrix) and want to transform into casual form (1D numeric series)
    '''
    return torch.argmax(y_image_matrix, dim=1).float()

@contextmanager
def timer(label='experiment'):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[{label}] Execution time: {execution_time:.6f} seconds")

def log_experiment(filepath, params, metrics, model, chart_path, description):
    exp_num = 1
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        matches = re.findall(r'## Experiment (\d+)', content)
        if matches:
            exp_num = max(int(m) for m in matches) + 1

    model_class_name = type(model).__name__

    def format_grouped_architecture(root_model):
        md_sections = []
        for child_name, child_module in root_model.named_children():
            class_name = type(child_module).__name__
            section_lines = [f"#### `{class_name}`:"]
            
            for layer_name, layer in child_module.named_children():
                layer_str = str(layer).replace('\n', '').replace('  ', ' ')
                section_lines.append(f"* `{layer_str}`")
                
            md_sections.append("\n".join(section_lines))
        return "\n\n".join(md_sections) if md_sections else "* `No submodules identified`"

    architecture_str = format_grouped_architecture(model)

    runtime = metrics.get('execution_time', 0.0)
    time_str = f"{int(runtime // 60)}m {runtime % 60:.2f}s" if runtime >= 60 else f"{runtime:.2f}s"

    md_entry = f"""
## Experiment {exp_num}

> {description}

### Params
* **model_class**: `{model_class_name}`
* **batch**: {params.get('batch_size')}
* **latent**: {params.get('latent_dim')}
* **epochs_target**: {params.get('epochs')}
* **epochs_completed**: {params.get('epochs_completed')}
* **device**: {params.get('device')}
* **dataset**: {params.get('harmonic')}

### Model Architecture
{architecture_str}

### Metrics
| Metric | Best | Final |
| :--- | :--- | :--- |
| **Train JSD** | {metrics.get('best_train_jsd'):.4f} | {metrics.get('final_train_jsd'):.4f} |
| **Val JSD** | {metrics.get('best_val_jsd'):.4f} | {metrics.get('final_val_jsd'):.4f} |
| **Val MASE** | {metrics.get('best_val_mase'):.4f} | {metrics.get('final_val_mase'):.4f} |
| **Val IoU** | {metrics.get('best_val_iou'):.4f} | {metrics.get('final_val_iou'):.4f} |

**Execution time**: {time_str}

### Training Charts
![train_history_{exp_num}]({'../' + chart_path})

---
"""
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(md_entry)
        
    print(f" Successfully saved entry to {filepath} as Experiment {exp_num} ({model_class_name})!")

def get_exp_num(exp_file):
    if os.path.exists(exp_file):
        with open(exp_file, 'r', encoding='utf-8') as f:
            matches = re.findall(r'## Experiment (\d+)', f.read())
            if matches:
                return max(int(m) for m in matches) + 1
    return 1