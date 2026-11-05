#include "sparse_gemm.h"

#include <stdlib.h>
#include <string.h>

void sparse_weights_destroy(sparse_weights *s)
{
    if (!s) return;
    free(s->block_col_ptr);
    free(s->block_row_idx);
    free(s->block_values);
    memset(s, 0, sizeof(*s));
}
