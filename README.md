# Eigen-Compress 📉✨
> **"Image Compression and Noise Reduction Using an Advanced Linear Algebra Pipeline"**

Eigen-Compress is an interactive, browser-based data science dashboard that transforms images into algorithmic mathematical factors to demonstrate exactly how **Singular Value Decomposition (SVD)** fundamentally governs data compression. 

Developed specifically to bridge the understanding between abstract Linear Algebra Eigenvectors and physical, hard-drive data footprint limits!

## Core Features
1. **Interactive Global SVD Reconstructor:** Upload any Image and dynamically sweep a $K$ constant slider across the vector limits natively to observe the tradeoff between data size and structural clarity. 
2. **The "JPEG Paradigm" Evolution Block-Mode:** Mathematically slice your picture natively into $16 \times 16$ macro-blocks! By mapping SVD to localized matrices instead of global space, you natively harness the identical data science mechanism employed by actual image codecs!
3. **Advanced Matrix Archiving (.npz):** Re-export the mathematical arrays into a custom numpy binary via our Entropy-Pipeline! Includes integrated **Min-Max Dynamic Scaling (Int8 Quantizations)** and **Delta Pre-Conditioning (DPCM differences)** to compress mathematics into phenomenally sub-100kb storage configurations!
4. **Binary Decompressor Tool:** Seamless standalone unpacker integrated to test decoding payloads deterministically to guarantee the mathematics!

## Installation & Setup

Before running, guarantee you have Python installed, and then initialize the localized virtual environment requirements.

```bash
# Clone the repository logic
cd Eigen-Compress

# Install the numeric dependencies (NumPy, Streamlit, Matplotlib, Pillow)
pip install -r requirements.txt
```

## How to Deploy the Platform

Eigen-Compress operates as an interactive local Web App using Streamlit's pipeline. To launch the User Interface, inject the following command locally:

```bash
python -m streamlit run app.py
```
*The command will automatically bind to `localhost` and launch an interactive URL into your default browser!*

---
### Explore The Mathematics!
If you are interested in what drives the algorithms under the hood, explore `architecture.md` deployed in this repo! It dissects the exact Big-O complexities and information theory limitations surrounding the project scope. 
