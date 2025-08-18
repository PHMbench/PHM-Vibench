
---

# Model Factory Module (Continue updating)

## 🎯 Purpose

The **Model Factory** is responsible for dynamically building and initializing neural network models. It acts as a centralized dispatcher that constructs a model instance based on configuration parameters, promoting a clean separation between model architecture and the main training pipeline. This modular approach allows for easy experimentation and the addition of new architectures without modifying the core training code.

---

## 📂 Module Structure

The factory is organized around a main entry point and several subdirectories, each containing different families of model architectures.

**Note**: Each module should have 'if __name__ == '__main__':' to test the module. like
```python
if __name__ == '__main__':
    # Testing the RandomPatchMixer class
    def test_random_patch_mixer():
        B = 2  # Batch size
        L_list = [1024, 2048]  # Variable sequence lengths
        C_list = [8, 3]   # Variable channel dimensions

        patch_size_L = 128   # Patch size along L dimension
        patch_size_C = 5   # Patch size along C dimension
        num_patches = 100   # Number of patches to sample
        output_dim = 16    # Output dimension after mixing
        f_s = 100  # Sampling frequency

        model = E_01_HSE(patch_size_L, patch_size_C, num_patches, output_dim, f_s)

        for C in C_list:
            for L in L_list:
                x = torch.randn(B, L, C)
                y = model(x)
                print(f'Input shape: {x.shape}, Output shape: {y.shape}')

    # Run the test
    test_random_patch_mixer()
```

| File / Directory     | Description                                                                                                                                                |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_factory.py` | The main entry file. It contains the core function `model_factory(args_model, metadata)` which dynamically imports and instantiates the requested model. |
| `MLP/`             | Contains Multi-Layer Perceptron-based models (e.g.,`Dlinear.py`).                                                                                        |
| `CNN/`              | Houses Convolutional Neural Network-based models (e.g., `ResNet.py`).                                                                                      |
| `RNN/`              | Contains Recurrent Neural Network-based models (e.g., `LSTM.py`).                                                                                         |
| `NO`           | Neural Operator-based models (e.g., `FNO.py`).                                                                                                                 |
| `Transformer/`     | Home for various Transformer-based architectures.                                                                                                          |
| `ISFM/`            | A specialized module for the "industrial signal foundation model" family, which includes its own sub-modules for embeddings, backbones, and task heads.    |
| `FewShot/`         | Architectures designed specifically for Few-Shot Learning tasks (e.g.,`ProtoNet.py`).                                                                    |
| `X_model/`         | A directory for XAI models.                                                                                                                                |

---

## ⚙️ Configuration

To specify which model to build, you need to populate the `model` section in your YAML configuration file. The factory uses these arguments to find and initialize the correct model class.

**Key Configuration Fields:**

* **`type`**: The name of the subdirectory where the model is located (e.g., `MLP`, `ISFM`).
* **`name`**: The name of the Python file (and the class within it) that defines the model (e.g., `Dlinear`).
* **`weights_path`** (Optional): A path to a pre-trained `.ckpt` file. If provided, the factory will load the saved weights into the model.
* **Other Parameters**: Any additional key-value pairs within the `model` section will be passed as hyperparameters to the model's constructor.

**Example Configuration (`.yaml`):**

```yaml
model:
  name: "M_01_ISFM"
  type: "ISFM"
  weights_path: False

  input_dim: 2
  


  num_heads: 4
  num_layers: 2

  d_ff: 128

  dropout: 0.1

  hidden_dim: 64
  activation: "relu"
  # ISFM_args:
  num_patches: 128
  embedding:  E_01_HSE # E_01_HTFE
  patch_size_L: 256
  patch_size_C: 1
  output_dim: 1024
  backbone: B_08_PatchTST # B_04_Dlinear , B_06_TimesNet, B_08_PatchTST,B_01_basic_transformer

  num_layers: 1

  task_head: H_09_multiple_task #
  classification_head: H_02_distance_cla
  prediction_head: H_03_Linear_pred
  e_layers: 2
  factor: 5


```

---

## 🌊 Workflow

The model instantiation process is straightforward:

1. **Read Configuration**: The training pipeline reads the `model` arguments from the YAML configuration file.
2. **Dynamic Import**: The `model_factory` uses the `type` and `name` fields to dynamically construct the import path (e.g., `src.model_factory.ISFM.M_01_ISFM`).
3. **Instantiation**: It imports the `Model` class from the specified file and initializes it by passing the entire `args_model` dictionary and the `metadata` object to its constructor.
   ```python
   # Inside model_factory.py
   model_module = importlib.import_module(f'.{args.type}.{args.name}', package='src.model_factory')
   model = model_module.Model(args, metadata)
   ```
4. **Load Checkpoint** (Optional): If `weights_path` is specified, the factory loads the parameters from the checkpoint file into the newly created model instance.
5. **Return Model**: The fully initialized model is returned.

---

## 🎁 Returned Object

The `model_factory` function returns a single object:

* **An initialized `torch.nn.Module` instance**: This model is ready to be used by the `Task_Factory` and the training pipeline.
