import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import io

st.set_page_config(page_title="Eigen-Compress: SVD Image Compression", layout="wide")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_mse(img1, img2):
    """Calculates Mean Squared Error between two image arrays."""
    return np.mean((img1 - img2) ** 2)

def quantize_factors(U, S, Vt):
    """
    Intelligently maps [-1.0, 1.0] factors into the full 8-bit dynamic range
    by applying Min-Max scaling, and hiding the reverse-scale factor directly inside 
    the Singular Values (Sigma), so the NPZ format remains geometrically identical!
    """
    max_U = np.max(np.abs(U))
    if max_U == 0: max_U = 1.0
    scale_U = 127.0 / max_U
    U_q = np.clip(np.round(U * scale_U), -127, 127).astype(np.int8)
    
    max_Vt = np.max(np.abs(Vt))
    if max_Vt == 0: max_Vt = 1.0
    scale_Vt = 127.0 / max_Vt
    Vt_q = np.clip(np.round(Vt * scale_Vt), -127, 127).astype(np.int8)
    
    # Absorb the scales into S so we don't need to save them separately
    S_new = (S / (scale_U * scale_Vt)).astype(np.float32)
    
    return U_q, S_new, Vt_q

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    # Since image is max 255
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def perform_svd(matrix):
    """Performs SVD and returns U, S, Vt."""
    # Using np.linalg.svd: matrix A = U * S * Vt
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt

def compress_matrix(U, S, Vt, k):
    """Reconstructs matrix using top k singular values."""
    # Reconstruct rank k matrix: U[:, :k] * np.diag(S[:k]) * Vt[:k, :]
    if len(S) == k and S.ndim == 1:
        # If already truncated during npz load
        reconstructed = np.dot(U, np.dot(np.diag(S), Vt))
    else:
        reconstructed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    # Clip values to valid image range [0, 255] and convert to uint8
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

# ==========================================
# MAIN APPLICATION
# ==========================================
st.title("Eigen-Compress: Image Compression via SVD")
st.markdown("Explore how Linear Algebra powers Data Compression by decomposing images into their fundamental structural components.")

tab1, tab2 = st.tabs(["🗜️ Compressor & Analyzer", "🔍 Binary .npz Decompressor"])

with tab1:
    # Image Loading
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        # Use default sample image
        sample_path = "sample_image.png"
        if os.path.exists(sample_path):
            image = Image.open(sample_path)
        else:
            # Generate a dummy noise image if sample not found
            image = Image.fromarray(np.random.randint(0, 256, (400, 400), dtype=np.uint8))
            st.warning("Sample image not found. Please upload an image.")
    else:
        image = Image.open(uploaded_file)

    # Mode Selection
    col_mode1, col_mode2 = st.columns([1, 1])
    with col_mode1:
        color_mode = st.radio("Select Processing Mode:", ["Grayscale (Default, Mathematical Foundation)", "Color (RGB Channel Splitting)"], index=0)
        is_color = "Color" in color_mode
    with col_mode2:
        st.info("Grayscale uses a single 2D matrix ($M \\times N$). Color splits the image into Red, Green, and Blue matrices and applies SVD to each.")

    # Process Image
    if is_color:
        image_array = np.array(image.convert("RGB")).astype(float)
        channels = [image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]]
        st.write(f"**Matrix Shape:** 3 channels of {channels[0].shape}. Total elements: {image_array.size}")
    else:
        image_array = np.array(image.convert("L")).astype(float)
        channels = [image_array]
        st.write(f"**Matrix Shape:** {image_array.shape[0]} rows x {image_array.shape[1]} columns. Total elements: {image_array.size}")

    st.divider()

    # Compute SVD for all channels
    svd_results = []
    for i, channel in enumerate(channels):
        U, S, Vt = perform_svd(channel)
        svd_results.append((U, S, Vt))

    # Extract the singular values of the first channel for analysis (representative)
    representative_S = svd_results[0][1]
    max_k = len(representative_S)

    # Determine reasonable default for k
    default_k = max(1, int(max_k * 0.1)) # Top 10%

    st.header("Step 1: Eigenvalue Energy Distribution (Scree Plot)")
    st.write("The Scree Plot visualizes the Singular Values in decreasing order. Notice how the curve drops sharply. This confirms that the vast majority of the image's 'structure' or 'energy' is captured by the first few principal components.")

    # Plot Scree Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(representative_S, marker='.', linestyle='-', color='b')
    ax.set_title("Magnitude of Singular Values")
    ax.set_xlabel(r"Component Index $i$")
    ax.set_ylabel(r"Singular Value $\sigma_i$")
    ax.set_xlim(0, min(max_k, 500)) # limit x axis to show drop clearly
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.divider()

    st.header("Step 2: Interactive Compression")
    st.write("Drag the slider to select the number of singular values $k$ to retain.")

    k = st.slider("Select number of Singular Values ($k$)", min_value=1, max_value=max_k, value=default_k, step=1)

    # Compress all channels
    compressed_channels = []
    for U, S, Vt in svd_results:
        comp_channel = compress_matrix(U, S, Vt, k)
        compressed_channels.append(comp_channel)

    if is_color:
        reconstructed_array = np.stack(compressed_channels, axis=-1)
    else:
        reconstructed_array = compressed_channels[0]

    # Metrics Calculation
    original_int = np.clip(image_array, 0, 255).astype(np.uint8)
    mse_val = calculate_mse(original_int.astype(float), reconstructed_array.astype(float))
    psnr_val = calculate_psnr(original_int.astype(float), reconstructed_array.astype(float))

    # Calculate storage footprint (approximation)
    original_values = image_array.size
    if is_color:
        M, N = channels[0].shape
        compressed_values = 3 * (M * k + k + k * N)
    else:
        M, N = image_array.shape
        compressed_values = M * k + k + k * N

    compression_ratio = compressed_values / original_values if original_values else 1

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Mean Squared Error (MSE)", f"{mse_val:.2f}", help="Lower means closer to original")
    col_metric2.metric("Peak Signal-to-Noise Ratio", f"{psnr_val:.2f} dB", help="Higher means better quality")
    col_metric3.metric("Data Size % (Original vs New)", f"{compression_ratio*100:.1f}%", help="Percentage of values needed to store the rank-k approximation compared to original matrix.")

    st.header("Step 3: Outcome Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        if is_color:
            st.image(original_int, use_container_width=True)
        else:
            st.image(original_int, use_container_width=True)

    with col2:
        st.subheader(f"Compressed Image (k={k})")
        if is_color:
            st.image(reconstructed_array, use_container_width=True)
        else:
            st.image(reconstructed_array, use_container_width=True)

    st.success("Comparison Mode explicitly shows that lower-rank approximation effectively 'decompresses' back into an image that looks almost identical to the original, eliminating redundancy.")

    st.divider()

    # STEP 4: Error Map
    st.header("Step 4: Error Map (Difference Image)")
    st.write("This heat map visualizes the mathematical information that was **discarded** during compression. Brighter areas indicate a higher loss of detail (edges, noise, textures).")

    diff_array = np.abs(original_int.astype(float) - reconstructed_array.astype(float))
    if is_color:
        diff_intensity = np.mean(diff_array, axis=-1)
    else:
        diff_intensity = diff_array

    fig_diff, ax_diff = plt.subplots(figsize=(8, 6))
    cax = ax_diff.imshow(diff_intensity, cmap='magma')
    ax_diff.axis('off')
    fig_diff.colorbar(cax, ax=ax_diff, label='Absolute Error (Pixel Intensity)')
    st.pyplot(fig_diff)

    st.divider()

    # DOWNLOADS AREA
    st.header("Export Compressed Data")
    st.write("Choose your export format. Exporting the Reconstructed PNG applies standard visual algorithms which may bloat in memory size. Exporting the Zipped Binary Archives (`.npz`) perfectly mirrors true spatial storage footprint savings!")
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        # Convert numpy array back to PIL Image
        if is_color:
            img_to_download = Image.fromarray(reconstructed_array, 'RGB')
        else:
            img_to_download = Image.fromarray(reconstructed_array, 'L')

        buf_png = io.BytesIO()
        img_to_download.save(buf_png, format="PNG")
        byte_png = buf_png.getvalue()

        st.download_button(
            label="🖼️ Download Reconstructed PNG",
            data=byte_png,
            file_name=f"compressed_k{k}.png",
            mime="image/png"
        )
        
    with col_dl2:
        # Save exact matrices to NPZ
        buf_npz = io.BytesIO()
        if is_color:
            U_r_q, S_r_q, Vt_r_q = quantize_factors(svd_results[0][0][:, :k], svd_results[0][1][:k], svd_results[0][2][:k, :])
            U_g_q, S_g_q, Vt_g_q = quantize_factors(svd_results[1][0][:, :k], svd_results[1][1][:k], svd_results[1][2][:k, :])
            U_b_q, S_b_q, Vt_b_q = quantize_factors(svd_results[2][0][:, :k], svd_results[2][1][:k], svd_results[2][2][:k, :])
            np.savez_compressed(
                buf_npz,
                mode=np.array(['color']),
                k=np.array([k]),
                U_r=U_r_q, S_r=S_r_q, Vt_r=Vt_r_q,
                U_g=U_g_q, S_g=S_g_q, Vt_g=Vt_g_q,
                U_b=U_b_q, S_b=S_b_q, Vt_b=Vt_b_q
            )
        else:
            U_q, S_q, Vt_q = quantize_factors(svd_results[0][0][:, :k], svd_results[0][1][:k], svd_results[0][2][:k, :])
            np.savez_compressed(
                buf_npz,
                mode=np.array(['grayscale']),
                k=np.array([k]),
                U=U_q, S=S_q, Vt=Vt_q
            )
        
        byte_npz = buf_npz.getvalue()
        st.download_button(
            label="📉 Download Zipped Binary Archive (.npz)",
            data=byte_npz,
            file_name=f"binary_archive_k{k}.npz",
            mime="application/octet-stream"
        )

with tab2:
    st.header("Upload & Reconstruct .npz Archives")
    st.write("Upload a Numpy Zipped Binary archive extracted from Tab 1. This viewer directly plots the physical spatial components ($U, \Sigma, V^T$) independently of PNG image rules.")
    
    upload_npz = st.file_uploader("Upload NPZ Binary", type=["npz"])
    
    if upload_npz is not None:
        try:
            with np.load(upload_npz) as data:
                mode = data['mode'][0]
                k_val = data['k'][0]
                
                st.info(f"Successfully loaded a **{mode}** archive. Embedded Rank Constraint: $k={k_val}$")
                
                if mode == 'color':
                    comp_r = compress_matrix(data['U_r'].astype(np.float32), data['S_r'], data['Vt_r'].astype(np.float32), k_val)
                    comp_g = compress_matrix(data['U_g'].astype(np.float32), data['S_g'], data['Vt_g'].astype(np.float32), k_val)
                    comp_b = compress_matrix(data['U_b'].astype(np.float32), data['S_b'], data['Vt_b'].astype(np.float32), k_val)
                    restored_image = np.stack([comp_r, comp_g, comp_b], axis=-1)
                else:
                    restored_image = compress_matrix(data['U'].astype(np.float32), data['S'], data['Vt'].astype(np.float32), k_val)

                st.subheader("Mathematically Reconstructed Frame")
                st.image(restored_image, use_container_width=True)
                
                st.divider()
                st.write("You can convert this decoded binary matrix array into a readable `.png` picture format underneath:")
                
                # PNG conversion
                buf_res = io.BytesIO()
                if mode == 'color':
                    img_res = Image.fromarray(restored_image, 'RGB')
                else:
                    img_res = Image.fromarray(restored_image, 'L')
                img_res.save(buf_res, format="PNG")
                
                st.download_button(
                    label="Export Decoded Output to PNG",
                    data=buf_res.getvalue(),
                    file_name="decoded_binary.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Error reading archive. Ensure this `.npz` file was generated from Tab 1. Details: {e}")
