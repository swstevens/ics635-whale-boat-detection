# Oahu Flood Impact Analysis: Multi-modal Satellite Segmentation with SatViT

This project demonstrates a complete applied AI workflow for mapping flood susceptibility on Oahu, Hawaii. The system utilizes **Sentinel-1 (SAR)** and **Sentinel-2 (Optical)** satellite imagery processed through a **Satellite Vision Transformer (SatViT)** encoder and a custom segmentation head.

### 📚 Sources and Citations
* **Original Dataset:** Training data was sourced from the **Sen1Floods11** dataset.  
    [https://github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
* **Model Architecture:** The core SatViT transformer implementation is based on the work by **antofuller**.  
    [https://github.com/antofuller/SatViT](https://github.com/antofuller/SatViT)

### 📥 Model Weights (External Storage)
Due to GitHub's 100MB file size limit, the pre-trained model weights (**~350MB**) are hosted on Google Drive. UH account required. 
* **Download Link:** [https://drive.google.com/file/d/1eTs08RpSIHsnn5nrwd354qOBhvf6-GbQ/view?usp=drive_link](https://drive.google.com/file/d/1eTs08RpSIHsnn5nrwd354qOBhvf6-GbQ/view?usp=drive_link)]
* **Download Link:** [https://drive.google.com/file/d/1Gaojkrm8rRK7fXBCirungBu2es8oPicl/view?usp=drive_link](https://drive.google.com/file/d/1Gaojkrm8rRK7fXBCirungBu2es8oPicl/view?usp=drive_link)
* **Download Link:** [https://drive.google.com/file/d/1zvD8JlA2Z1hGk_G59yCLsPwFO9LvYhZ8/view?usp=drive_link](https://drive.google.com/file/d/1zvD8JlA2Z1hGk_G59yCLsPwFO9LvYhZ8/view?usp=drive_link)]

**Instructions:** Download `final_project_encoder.pth`,`final_project_decoder.pth`, and `SatViT_V2.pt` and place them in the root of the `ICS605_final_streamlit_app/` directory, and `SatViT_V2.pt` in the ICS605_final_SatViT_model_train directory.

### 📂 Folder Structure
The repository is organized into two primary components: the training environment and the deployment dashboard.

```text
ics605-sp26-final-project-behrens-j/
├── requirements.txt  # Dependency list for environment setup
├── ICS605_final_SatViT_model_train/        # Model Development & Evaluation
│   ├── custom_flood_SatViT_train_test.ipynb # Training, mIoU evaluation, & failure analysis
│   ├── SatViT_inference.py                 # Training-specific inference scripts
│   ├── models.py                           # Model definitions used for training
│   ├── dataset.py                          # Data loading for training
│   ├── SatViT_V2.pt                        # Training checkpoint
│   └── SatViT/                             # Training-side architecture dependencies
│
└── ICS605_final_streamlit_app/             # Live Demo Application
    ├── streamlit_app.py                    # Main dashboard entry point       
    ├── backend/                            # Production AI pipeline logic
    │   ├── SatViT_inference.py             # Validation, fixing, and prediction logic
    │   ├── models.py                       # Model head and architecture definitions
    │   └── dataset.py                      # Production data utilities
    ├── final_project_decoder.pth           # Save Tweaked SatVit model savestate
    ├── final_project_encoder.pth           # Save segmentation decoder model savestate
    └── SatViT/                             # Core transformer model dependencies
```
### 🛠️ Environment Setup
This project utilizes `uv` for dependency management. To set up the environment and install the required packages (PyTorch, Rasterio, Streamlit, etc.):

```bash
# Create the virtual environment
uv venv

# Activate the environment
.venv\Scripts\activate

# Install dependencies from the requirements list
uv pip install -r requirements.txt
```

### 🚀 Running the Live Demo
The primary interface for this project is a Streamlit app that takes raw satellite imagery as input and produces high-resolution output visualizations[cite: 17, 18]. 

To launch the dashboard locally:
1. Navigate to the app directory: `cd ICS605_final_streamlit_app`
2. Run the application: `streamlit run streamlit_app.py`

## 🌍 Demo Data: Multi-Region Validation (Oahu)

To demonstrate the robustness of the **SatViT Flood Prediction Model**, I extracted a diverse dataset of 30 populated regions across Oahu. This ensures the model is evaluated against various environments, including dense urban corridors (Honolulu), rural coastal towns (Haleiwa), and mountainous residential areas (Manoa).

### 🛰️ Dataset Specifications
Each sample in this dataset was custom-processed via Google Earth Engine to meet the strict input requirements of the transformer architecture:
* **Dimensions:** 256 x 256 pixels (calculated via a 2,560m bounding box).
* **Resolution:** 10 meters per pixel.
* **Channels (Sentinel-2):** 13-band Multi-spectral (B1–B12) harmonized imagery.
* **Channels (Sentinel-1):** Dual-polarization Radar (VV + VH) to penetrate cloud cover.

### 📂 Accessing the Imagery
The processed `.tif` files used in the live demo are hosted on Google Drive. These files are ready to be uploaded directly into the Streamlit interface for real-time inference.

> **[Download Oahu Demo Dataset (30 Regions)](https://drive.google.com/drive/folders/1Pp3_-HUiSLH-7740DWg3Hlq9h413q2or?usp=drive_link)**

### 🛠️ How to use in the Demo
1.  **Download** the `.zip` file from the link above and extract the images.
2.  **Launch** the Streamlit app.
3.  **Upload** a Sentinel-2 (Optical) `.tif` and its corresponding Sentinel-1 (Radar) `.tif`.
4.  **Observe** the side-by-side visualization showing the original satellite view vs. the model's predicted flood vulnerability mask.

### 📍 Featured Locations in Demo
| Region Type | Included Samples |
| :--- | :--- |
| **Urban Hubs** | Honolulu Downtown, Waikiki, Pearl City, Kapolei |
| **North Shore** | Haleiwa, Waialua, Kahuku |
| **Windward Side** | Kaneohe, Kailua, Waimanalo |
| **Leeward Coast** | Waianae, Makaha, Nanakuli |

### 📊 Training, Results, and Limitations
The `custom_flood_SatViT_train_test.ipynb` notebook provides a technical record of the model build and evaluation. 

* **Success Case:** Demonstrates clear identification of flood zones in high-confidence Oahu regions.
* **Limitations & Failure Cases:** Model performance issues, such as interference from extreme cloud cover or urban shadows.
