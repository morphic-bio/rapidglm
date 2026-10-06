#include <rapidglm/rapidglm_c.h>
#include <stdio.h>

int main(void) {
    /* Poisson fit of one response on an intercept and a group indicator. */
    const double design[] = {1, 0, 1, 0, 1, 1, 1, 1};
    const size_t column_ptr[] = {0, 4}, row_index[] = {0, 1, 2, 3};
    const double counts[] = {2, 3, 8, 9};
    rapidglm_dense x = {4, 2, design};
    rapidglm_sparse y = {4, 1, column_ptr, row_index, counts};
    rapidglm_fit_options options;
    rapidglm_table *table = NULL;
    rapidglm_fit_options_init(&options);
    options.family = RAPIDGLM_POISSON;
    if (rapidglm_fit_batch(&x, &y, &options, 1, &table) != RAPIDGLM_OK) {
        fprintf(stderr, "rapidglm: %s\n", rapidglm_last_error());
        return 1;
    }
    printf("RapidGLM %s C API %d: beta1 = %.6f\n", rapidglm_version(), rapidglm_c_api_version(),
           rapidglm_table_numeric(table, "beta1")[0]);
    rapidglm_table_free(table);
    return 0;
}
