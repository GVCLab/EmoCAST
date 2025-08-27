# EmoCAST: Emotional Talking Portrait via Emotive Text Description

[![Project Website](https://img.shields.io/badge/Project-Website-Green)](https:)
[![arXiv](https://img.shields.io/badge/ArXiv--red)](https://arxiv.org/abs/)

![teaser](assets/teaser.png)

We introduce **EmoCAST**, a novel diffusion-based emotional talking head system for in-the-wild images that incorporates flexible and customizable emotive text prompts, as shown in the example.

## 📸 Demo

## 🔧️ Framework

![method](assets/method.png)

## ⚙️ Usage
### 🛠️ Installation

1. Create conda environment
   ```bash
   conda create -n emocast python=3.10
   conda activate emocast
   ```

2. Install packages with pip

   ```bash
   pip install -r requirements.txt
   pip install .
   ```

### 🎮 Inference

1. Download Pretrained Models
   
   Download these models below into the `./pretrained_model/` folder.
  
    |  Model | Download Link |   
    |:--------:|:------------:|
    |audio_separator |  [https://huggingface.co/huangjackson/Kim_Vocal_2](https://huggingface.co/huangjackson/Kim_Vocal_2)|
    |insightface | [https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo)|
    |face landmarker | [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |
    |motion module | [https://github.com/guoyww/AnimateDiff/blob/main/README.md#202309-animatediff-v2](https://github.com/guoyww/AnimateDiff/blob/main/README.md#202309-animatediff-v2)|
    |sd-vae-ft-mse | [https://huggingface.co/stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)|
    |StableDiffusion V1.5|[https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)|
    |wav2vec | [https://huggingface.co/facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)|
    |EmoCAST||

    These pretrained models should be organized as follows:

    ```text
    ./pretrained_models/
    |-- audio_separator/
    |   |-- download_checks.json
    |   |-- mdx_model_data.json
    |   |-- vr_model_data.json
    |   |-- Kim_Vocal_2.onnx
    |-- face_analysis/
    |   |-- models/
    |       |-- face_landmarker_v2_with_blendshapes.task  # face landmarker model from mediapipe
    |       |-- 1k3d68.onnx
    |       |-- 2d106det.onnx
    |       |-- genderage.onnx
    |       |-- glintr100.onnx
    |       |-- scrfd_10g_bnkps.onnx
    |-- motion_module/
    |   |-- mm_sd_v15_v2.ckpt
    |-- sd-vae-ft-mse/
    |   |-- config.json
    |   |-- diffusion_pytorch_model.safetensors
    |-- stable-diffusion-v1-5/
    |   |-- unet/
    |       |-- config.json
    |       |-- diffusion_pytorch_model.safetensors
    |-- wav2vec/
        |-- wav2vec2-base-960h/
            |-- config.json
            |-- feature_extractor_config.json
            |-- model.safetensors
            |-- preprocessor_config.json
            |-- special_tokens_map.json
            |-- tokenizer_config.json
            |-- vocab.json
    ```

2. Prepare Inference Data

   Prepare the *driving audio*, *reference image*, and *emotive text prompt* as input, and change the `--driving_audio` and `--source_image` to the correct path.

   - For the *driving audio*, it should be in `.wav` format.

   - For the *reference image*, it should be cropped into a square with the face as the primary focus, facing forward.
  
   - For the *emotive text prompt*, it should describe a specific talking scene, such as:
     
     `The portrait is experiencing chronic illness or pain.`
     
     `A person is talking with happy emotion.`

     `The portrait is watching a horror movie with jump scares.`
   
   Here, we provide [some samples](examples/) for your reference.


3. Run Inference
   ```bash
   
   ```

## 🛎 Citation
If you find our work helpful for your research, please cite:
```

```

## 💗 Acknowledgements
