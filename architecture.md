# Architecture of Eigen-Compress

This document outlines the underlying mechanisms of the **Eigen-Compress** application, which reduces the size of image data utilizing **Singular Value Decomposition (SVD)**.

## 1. Mathematical Foundation

### Matrix Representation
An image can be interpreted mathematically as a continuous flow of data points organized in a 2D matrix $A$ of size $M \times N$.
- In Grayscale, each matrix cell $A_{i,j}$ contains a single pixel intensity value from 0 to 255.
- In Color (RGB), the application splits the image into three separate matrices representing the Red, Green, and Blue channels respectively, before applying mathematical operations to each independently.

### Singular Value Decomposition (SVD)
The core algebraic theorem applied is SVD, which states that any rectangular real-valued matrix $A$ can be factored into three unique matrices:
$$ A_{M \times N} = U_{M \times M} \cdot \Sigma_{M \times N} \cdot V^T_{N \times N} $$

- **$U$**: An orthogonal matrix whose columns act as the "basis vectors" for the column space. These represent structural features.
- **$\Sigma$**: A diagonal matrix containing the **Singular Values** ($\sigma_i$) in decreasing mathematical magnitude. 
- **$V^T$**: An orthogonal matrix whose rows act as the mathematical basis vectors for the row space. 

By calculating the SVD, we transpose the image's raw pixel data into a collection of completely independent structural elements, sorted by their "energy" (represented by the size of the singular values). 

## 2. Dimensionality Reduction & Compression

If an image is largely composed of smooth textures with distinct edges, a few massive "dominant patterns" dictate the image, and the rest is micro-noise or minor artifacts with very small singular values.

**Rank-$k$ Approximation:**
Instead of storing all the singular values and complete basis arrays, the application enforces a hard rank limit, $k$, effectively truncating the mathematical precision mapping:
$$ A_k = U_{ :, 1:k } \cdot \Sigma_{ 1:k, 1:k } \cdot V^T_{ 1:k, : } $$

### The Big-O Impact (Data Compression)
For a $M \times N$ matrix, we normally store $M \cdot N$ pixel values.
Using Rank-$k$ approximation to store the compressed representation, we only store:
- $k$ columns of $U$ = $M \cdot k$ values
- $k$ singular values of $\Sigma$ = $k$ values
- $k$ rows of $V^T$ = $k \cdot N$ values

Total footprint: $k(M + 1 + N)$. 
When $k \ll \min(M, N)$, $k(M + 1 + N)$ is substantially smaller than $M \cdot N$. This is where the compression occurs, dramatically reducing computational byte size.

### Physical vs File Format Compression
An important realization: Reconstructing the matrix into a classical spatial grid to save it as a `.png` format does *not* save space physically. In fact, PNG uses continuous-tone lossless compression (DEFLATE), meaning the newly introduced "SVD artifact noise" makes the file *harder* to shrink, causing the PNG filesize to expand!

To prove true mathematical footprint reduction on a hard-drive, the application exports the factor arrays directly into an algorithmic `.npz` Zipped Binary Archive. Bypassing pixel grids, these binaries directly mirror the $k(M + 1 + N)$ space footprint rule.

## 3. Application Flow 

1. **User Input Phase:** A raw image is submitted and pre-processed in Python natively via the PIL (Pillow) library.
2. **Channel Strategy:** The data is piped as 2D NumPy arrays. If RGB is active, three distinct $M \times N$ matrices are created. 
3. **Array Factorization (`np.linalg.svd`)**: NumPy engages LAPACK C-Routines in its backend to iteratively solve for the real Eigenvalues bounding the $AA^T$ projection mapping.
4. **Selective Reconstruction**: Controlled via the user slider ($k$), the application linearly multiplies the dimension-reduced sub-blocks back together iteratively applying $\min\max(0, 255)$ bound clipping techniques.
5. **Loss Metrics & Visualization**: The Error Map (Noise matrix) isolates what mathematical features were discarded, while standard metrics like MSE (Mean Squared Error) explicitly compare matrix variance shifts.

*Thus, Eigen-Compress proves how linear algebra bridges abstract coordinate representations strictly into data-center operational limits.*
