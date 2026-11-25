#include "sparse_gemm.h"

// Row tile: four input rows
#define TM 4

// helper function to handle the leftover input rows
// we cannot handle through tiling --> single-row scalar sparse GEMM
static void sparse_row_scalar(const float *a_row, const sparse_weights *w,
                              float *c_row) {
    for (size_t g = 0; g < w->n / 4; ++g) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
            size_t k0 = 4 * (size_t)w->block_row_idx[p];
            for (size_t r = 0; r < 4; ++r) {
                float av = a_row[k0 + r];
                for (size_t col = 0; col < 4; ++col)
                    acc[col] += av * w->block_values[16 * p + 4 * r + col];
            }
        }
        for (size_t col = 0; col < 4; ++col)
            c_row[4 * g + col] = acc[col];
    }
}

int gemm_sparse_tiled(const float *a, const sparse_weights *w, float *c,
                      size_t m) {
    // check pointer are not null
    if (!a || !w || !c)
        return SPARSE_ERROR;

    /*
    obtain input columns (k),
    weight columns (n),
    number of weight groups (4 weight columns),
    how many input rows can be handles through tiles --> the
    leftover rows/columns are handles separately
    */
    const size_t k = w->k;
    const size_t n = w->n;
    const size_t groups = n / 4;
    const size_t m_full = m - m % TM;

    // iterate on the INPUT tiles, 4 WHOLE INPUT rows
    for (size_t i0 = 0; i0 < m_full; i0 += TM) {
        const float *a0 = a + (i0 + 0) * k;
        const float *a1 = a + (i0 + 1) * k;
        const float *a2 = a + (i0 + 2) * k;
        const float *a3 = a + (i0 + 3) * k;

        // iterate on weight groups --> select 4 WHOLE WEIGHT columns
        // initialise the 16 output sums to 0 (4 input rows x 4 output columns)
        for (size_t g = 0; g < groups; ++g) {
            float acc[TM][4] = {{0.0f}};

            // for each group select its start and end (excluded)
            // iterate from start to end --> p: weight subgroup idx
            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                /*
                get tile p row idx
                obtain k0: first row idx of this block in the full weight matrix -->
                determines which input values to use
                *w_blk: pointer to the first stored
                value in tile p --> determines where to read the tile’s weights
                */
                const size_t k0 = 4 * (size_t)w->block_row_idx[p];
                const float *w_blk = w->block_values + 16 * p;

                /*
                now you identified: the input whole 4 rows AND the weight subgroup
                iterate on the rows of the weight subgroup and access each value in the
                subgroup row AND the matching input values --> perform the matrix
                product fixing the weight row and access the input values in the
                matching column
                */
                for (size_t r = 0; r < 4; ++r) {
                // obtain weight row values
                const float w0 = w_blk[4 * r + 0];
                const float w1 = w_blk[4 * r + 1];
                const float w2 = w_blk[4 * r + 2];
                const float w3 = w_blk[4 * r + 3];

                // obtain input column values
                // for each input row select the single matching value
                // --> together these 4 values form one input tile column
                const float x0 = a0[k0 + r];
                const float x1 = a1[k0 + r];
                const float x2 = a2[k0 + r];
                const float x3 = a3[k0 + r];

                // perform all multiplications and cumulative sums
                // each input row has its own output sums --> do not sum rows together
                // data reuse: same weights reused across the 4 input rows
                acc[0][0] += x0 * w0;
                acc[0][1] += x0 * w1;
                acc[0][2] += x0 * w2;
                acc[0][3] += x0 * w3;

                acc[1][0] += x1 * w0;
                acc[1][1] += x1 * w1;
                acc[1][2] += x1 * w2;
                acc[1][3] += x1 * w3;

                acc[2][0] += x2 * w0;
                acc[2][1] += x2 * w1;
                acc[2][2] += x2 * w2;
                acc[2][3] += x2 * w3;

                acc[3][0] += x3 * w0;
                acc[3][1] += x3 * w1;
                acc[3][2] += x3 * w2;
                acc[3][3] += x3 * w3;
                }
            }

            // copy results into the full output matrix c
            // iterate on tile rows and tile columns
            for (size_t t = 0; t < TM; ++t) {
                for (size_t col = 0; col < 4; ++col) {
                /*
                select correct row:
                i0 is the first selected input row --> same starting row in the output
                move to row i0+t --> +t
                actually reach row i0+t by skipping the previous rows --> *n
                (n = number of output columns, each full row has n elements)

                select correct column: select
                select the correct output group of 4 columns
                and get its first column idx --> + 4*g
                select the single column within that group --> +col
                */
                c[(i0 + t) * n + 4 * g + col] = acc[t][col];
                }
            }

            /* iterate on the remaining values we cannot handle with tiling
            use scalar sparse GEMM:
                address of the start of input row i
                address of the start of output row i
                sparse weights
            */
            }
        }
    for (size_t i = m_full; i < m; ++i)
        sparse_row_scalar(a + i * k, w, c + i * n);

    return SPARSE_OK;
}