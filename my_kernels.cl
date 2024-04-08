// Calculating intensity histogram
__kernel void histogram(__global const uchar* A, __global uint* hist) {
    int id = get_global_id(0);
    atomic_inc(&hist[A[id]]);
}

// Calculating cumulative histogram
__kernel void cumulativeHistogram(__global const uint* hist, __global uint* cumHist) {
    int id = get_global_id(0);
    uint sum = 0;
    for (int i = 0; i <= id; ++i) {
        sum += hist[i];
    }
    cumHist[id] = sum;
}

// Normalizing and scaling cumulative histogram
__kernel void normalizeScaleHistogram(__global const uint* cumHist, __global uchar* normScaledHist) {
    int id = get_global_id(0);
    uint maxVal = cumHist[255];
    normScaledHist[id] = (uchar)((cumHist[id] * 255) / maxVal);
}

// Back-projection for histogram equalization
__kernel void backProjection(__global const uchar* A, __global const uchar* normScaledHist, __global uchar* B) {
    int id = get_global_id(0);
    B[id] = normScaledHist[A[id]];
}

// Calculating intensity histogram (16-bit)
__kernel void histogram16(__global const ushort* A, __global uint* hist) {
    int id = get_global_id(0);
    atomic_inc(&hist[A[id]]);
}

// Calculating cumulative histogram (16-bit)
__kernel void normalizeScaleHistogram16bit(__global const uint* cumHist, __global ushort* normScaledHist) {
    int id = get_global_id(0);

    // Find the maximum value in the cumulative histogram
    uint maxVal = 0;
    for (int i = 0; i <= 65535; ++i) {
        maxVal = max(maxVal, cumHist[i]);
    }

    // Normalize and scale the cumulative histogram
    normScaledHist[id] = (ushort)((cumHist[id] * 65535) / maxVal);
}

// Normalizing and scaling cumulative histogram (16-bit)
__kernel void normalizeScaleHistogram16(__global const uint* cumHist, __global ushort* normScaledHist) {
    int id = get_global_id(0);
    uint maxVal = cumHist[65535];
    normScaledHist[id] = (ushort)((cumHist[id] * 65535) / maxVal);
}

// Back-projection (histogram equalization) (16-bit)
__kernel void backProjection16(__global const ushort* A, __global const ushort* normScaledHist, __global ushort* B) {
    int id = get_global_id(0);
    B[id] = normScaledHist[A[id]];
}
