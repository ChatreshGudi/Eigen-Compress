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
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0: return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def blockify(img, B):
    """Pads and reshapes an M x N image into a stack of B x B blocks."""
    pad_h = (B - img.shape[0] % B) % B
    pad_w = (B - img.shape[1] % B) % B
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    H, W = padded.shape
    blocks = padded.reshape(H//B, B, W//B, B).transpose(0, 2, 1, 3).reshape(-1, B, B)
    return blocks, (H, W), (pad_h, pad_w)

def unblockify(blocks, shape, padding):
    """Restores stacked B x B blocks back into the M x N image dimensions."""
    H, W = shape
    pad_h, pad_w = padding
    B = blocks.shape[-1]
    img = blocks.reshape(H//B, W//B, B, B).transpose(0, 2, 1, 3).reshape(H, W)
    if pad_h > 0 or pad_w > 0:
        img = img[:H-pad_h, :W-pad_w]
    return img

def delta_encode(matrix, axis):
    """Pre-conditions 2D or 3D signals into zero-heavy deltas."""
    deltas = np.diff(matrix.astype(int), axis=axis).astype(np.int8)
    if matrix.ndim == 2:
        if axis == 0: return np.vstack((matrix[0:1, :], deltas))
        else: return np.hstack((matrix[:, 0:1], deltas))
    elif matrix.ndim == 3:
        if axis == 1: return np.concatenate((matrix[:, 0:1, :], deltas), axis=1)
        else: return np.concatenate((matrix[:, :, 0:1], deltas), axis=2)

def delta_decode(matrix, axis):
    """Restores the exact matrix from its delta map via two's complement cumsum."""
    return np.cumsum(matrix, axis=axis, dtype=np.int8)

def quantize_factors(U, S, Vt):
    """Min-Max Dynamic Scaling quantization (Works for 2D and 3D)."""
    max_U = np.max(np.abs(U))
    if max_U == 0: max_U = 1.0
    scale_U = 127.0 / max_U
    U_q = np.clip(np.round(U * scale_U), -127, 127).astype(np.int8)
    
    max_Vt = np.max(np.abs(Vt))
    if max_Vt == 0: max_Vt = 1.0
    scale_Vt = 127.0 / max_Vt
    Vt_q = np.clip(np.round(Vt * scale_Vt), -127, 127).astype(np.int8)
    
    # Encode with delta logic based on dimensions
    u_axis = 0 if U_q.ndim == 2 else 1
    vt_axis = 1 if Vt_q.ndim == 2 else 2
    
    U_enc = delta_encode(U_q, axis=u_axis)
    Vt_enc = delta_encode(Vt_q, axis=vt_axis)
    
    S_new = (S / (scale_U * scale_Vt)).astype(np.float32)
    return U_enc, S_new, Vt_enc

def compress_matrix(U, S, Vt, k, shape=None, padding=None):
    """Dynamically routes 2D algebraic matrix ops vs 3D Batch Vectorized ops."""
    if U.ndim == 2:
        S_trunc = S[:k] if S.ndim == 1 else S
        reconstructed = np.dot(U[:, :k], np.dot(np.diag(S_trunc), Vt[:k, :]))
        return np.clip(reconstructed, 0, 255).astype(np.uint8)
    else:
        # Vectorized 3D matrix multiplication
        S_exp = S[:, np.newaxis, :k] if len(S.shape) == 2 else S[:, np.newaxis, :]
        U_S = U[:, :, :k] * S_exp
        reconstructed_blocks = np.matmul(U_S, Vt[:, :k, :])
        reconstructed_blocks = np.clip(reconstructed_blocks, 0, 255)
        return unblockify(reconstructed_blocks, shape, padding).astype(np.uint8)

# ==========================================
# MAIN APPLICATION
# ==========================================
st.title("Eigen-Compress: Image Compression via SVD")
st.markdown("Explore how Linear Algebra powers Data Compression by decomposing images into their fundamental structural components.")

tab1, tab2 = st.tabs(["🗜️ Compressor & Analyzer", "🔍 Binary .npz Decompressor"])

with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        sample_path = "sample_image.png"
        image = Image.open(sample_path) if os.path.exists(sample_path) else Image.fromarray(np.random.randint(0, 256, (400, 400), dtype=np.uint8))
    else:
        image = Image.open(uploaded_file)

    col_mode1, col_mode2 = st.columns([1, 1])
    with col_mode1:
        color_mode = st.radio("Color Space:", ["Grayscale", "Color (RGB)"], index=0, horizontal=True)
        is_color = "Color" in color_mode
        algo_mode = st.radio("SVD Algorithm Architecture:", ["Global Matrix SVD", "Block-Based SVD (JPEG Paradigm)"], index=0)
        is_blocked = "Block-Based" in algo_mode
    
    with col_mode2:
        st.info("The JPEG Paradigm chunks the image into independent miniature blocks rather than forcing vectors to span across the entire image mathematically. This results in unprecedented file reduction since localized sub-textures exhibit far less geometric chaos!")
        if is_blocked:
            B = st.selectbox("Block Dimensions (B x B)", [8, 16, 32], index=1)
        else:
            B = 1 # dummy for footprint

    # Process Array Structure
    if is_color:
        image_array = np.array(image.convert("RGB")).astype(float)
        channels = [image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]]
    else:
        image_array = np.array(image.convert("L")).astype(float)
        channels = [image_array]

    st.divider()

    # Dynamic Factorization
    svd_results = []
    with st.spinner("Calculating matrices..."):
        if is_blocked:
            for channel in channels:
                blocks, s_shape, s_pad = blockify(channel, B)
                U, S, Vt = np.linalg.svd(blocks, full_matrices=False)
                svd_results.append({'U': U, 'S': S, 'Vt': Vt, 'shape': s_shape, 'padding': s_pad})
            max_k = B
        else:
            for channel in channels:
                U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                svd_results.append({'U': U, 'S': S, 'Vt': Vt, 'shape': None, 'padding': None})
            max_k = len(svd_results[0]['S'])

    # Determine default K
    default_k = max(1, int(max_k * 0.1)) if not is_blocked else max(1, int(B * 0.25))

    st.header("Interactive Parameters")
    st.write(f"Adjust $K$. Maximum limit is restricted computationally to {max_k}.")
    k = st.slider("Select Rank Truncation ($k$)", min_value=1, max_value=max_k, value=default_k, step=1)

    # Reconstruct Image
    compressed_channels = []
    for res in svd_results:
        comp_ch = compress_matrix(res['U'], res['S'], res['Vt'], k, res['shape'], res['padding'])
        compressed_channels.append(comp_ch)

    reconstructed_array = np.stack(compressed_channels, axis=-1) if is_color else compressed_channels[0]
    original_int = np.clip(image_array, 0, 255).astype(np.uint8)

    # Metrics
    mse_val = calculate_mse(original_int.astype(float), reconstructed_array.astype(float))
    psnr_val = calculate_psnr(original_int.astype(float), reconstructed_array.astype(float))

    original_values = image_array.size
    if is_blocked:
        num_blocks = svd_results[0]['U'].shape[0]
        channel_vals = num_blocks * (B * k + k + k * B)
        compressed_values = channel_vals * 3 if is_color else channel_vals
    else:
        M, N = channels[0].shape
        channel_vals = M * k + k + k * N
        compressed_values = channel_vals * 3 if is_color else channel_vals

    compression_ratio = compressed_values / original_values if original_values else 1

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Mean Squared Error (MSE)", f"{mse_val:.2f}")
    col_metric2.metric("PSNR (Quality)", f"{psnr_val:.2f} dB")
    col_metric3.metric("Mathematical Payload Size", f"{compression_ratio*100:.1f}%")

    st.header("Outcome Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_int, use_container_width=True)
    with col2:
        st.subheader(f"Compressed Image (k={k})")
        st.image(reconstructed_array, use_container_width=True)

    st.divider()
    
    st.header("Error Map (Difference Target)")
    diff_array = np.abs(original_int.astype(float) - reconstructed_array.astype(float))
    diff_intensity = np.mean(diff_array, axis=-1) if is_color else diff_array
    fig_diff, ax_diff = plt.subplots(figsize=(8, 6))
    cax = ax_diff.imshow(diff_intensity, cmap='magma')
    ax_diff.axis('off')
    fig_diff.colorbar(cax, ax=ax_diff, label='Absolute Error')
    st.pyplot(fig_diff)

    st.divider()

    st.header("Export Compressed Data")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        img_to_download = Image.fromarray(reconstructed_array, 'RGB' if is_color else 'L')
        buf_png = io.BytesIO()
        img_to_download.save(buf_png, format="PNG")
        st.download_button(label="🖼️ Download Reconstructed PNG", data=buf_png.getvalue(), file_name=f"c_k{k}.png", mime="image/png")
        
    with col_dl2:
        buf_npz = io.BytesIO()
        # Build dictionary dynamically based on mode and channels
        npz_dict = {
            'mode': np.array(['color' if is_color else 'grayscale']),
            'k': np.array([k]),
            'algo': np.array(['block' if is_blocked else 'global'])
        }
        if is_blocked:
            npz_dict['shape'] = np.array(svd_results[0]['shape'])
            npz_dict['padding'] = np.array(svd_results[0]['padding'])
            npz_dict['B'] = np.array([B])

        prefix = ['r', 'g', 'b'] if is_color else ['gray']
        for idx, pref in enumerate(prefix):
            r = svd_results[idx]
            U_q, S_q, Vt_q = quantize_factors(r['U'][..., :k], r['S'][..., :k], r['Vt'][..., :k, :])
            npz_dict[f'U_{pref}'] = U_q
            npz_dict[f'S_{pref}'] = S_q
            npz_dict[f'Vt_{pref}'] = Vt_q

        np.savez_compressed(buf_npz, **npz_dict)
        st.download_button(label="📉 Download Zipped Archive (.npz)", data=buf_npz.getvalue(), file_name=f"b_k{k}.npz", mime="application/octet-stream")

with tab2:
    st.header("Upload & Decode .npz Archive")
    upload_npz = st.file_uploader("Upload NPZ Binary", type=["npz"])
    if upload_npz is not None:
        try:
            with np.load(upload_npz) as data:
                mode = data['mode'][0]
                k_val = data['k'][0]
                algo = data['algo'][0] if 'algo' in data else 'global'
                
                st.info(f"Loaded Archive | Space: {mode} | Base: {algo} | K={k_val}")
                
                shape = data['shape'] if 'shape' in data else None
                padding = data['padding'] if 'padding' in data else None
                
                def decode_channel(pref):
                    U_enc, S_enc, Vt_enc = data[f'U_{pref}'], data[f'S_{pref}'], data[f'Vt_{pref}']
                    u_ax = 0 if U_enc.ndim == 2 else 1
                    vt_ax = 1 if Vt_enc.ndim == 2 else 2
                    U = delta_decode(U_enc, axis=u_ax).astype(np.float32)
                    Vt = delta_decode(Vt_enc, axis=vt_ax).astype(np.float32)
                    return compress_matrix(U, S_enc, Vt, k_val, shape=shape, padding=padding)

                if mode == 'color':
                    restored_image = np.stack([decode_channel('r'), decode_channel('g'), decode_channel('b')], axis=-1)
                else:
                    restored_image = decode_channel('gray')

                st.subheader("Mathematically Reconstructed Frame")
                st.image(restored_image, use_container_width=True)
                
                buf_res = io.BytesIO()
                Image.fromarray(restored_image, 'RGB' if mode=='color' else 'L').save(buf_res, format="PNG")
                st.download_button("Export Decoded Output to PNG", data=buf_res.getvalue(), file_name="decoded_output.png", mime="image/png")
        except Exception as e:
            st.error(f"Archive processing failure. Error: {e}")
