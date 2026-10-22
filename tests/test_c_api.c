/* Pure C use of the RapidGLM C interface: builds as C99 and checks that every
 * entry point runs and returns sensible tables. */
#include <math.h>
#include <rapidglm/rapidglm_c.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "%s:%d: check failed: %s (%s)\n", __FILE__, __LINE__, #condition,      \
                    rapidglm_last_error());                                                        \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

/* Eight samples in two groups; 20 genes, the first five changed fourfold. */
enum { SAMPLES = 8, GENES = 20 };
static double design[SAMPLES * 2];
static size_t column_ptr[GENES + 1], row_index[SAMPLES * GENES];
static double values[SAMPLES * GENES];

static void build(void) {
    size_t k = 0;
    unsigned state = 12345;
    for (size_t i = 0; i < SAMPLES; ++i) {
        design[2 * i] = 1;
        design[2 * i + 1] = i >= SAMPLES / 2;
    }
    column_ptr[0] = 0;
    for (size_t j = 0; j < GENES; ++j) {
        for (size_t i = 0; i < SAMPLES; ++i) {
            double mean = 20 + 10 * (double)(j % 7);
            if (j < 5 && i >= SAMPLES / 2)
                mean *= 4;
            state = state * 1103515245u + 12345u;
            double noise = 0.8 + 0.4 * (double)((state >> 16) % 1000) / 1000.0;
            double v = floor(mean * noise);
            if (v > 0) {
                row_index[k] = i;
                values[k++] = v;
            }
        }
        column_ptr[j + 1] = k;
    }
}

int main(void) {
    rapidglm_dense x = {SAMPLES, 2, design};
    rapidglm_sparse counts;
    rapidglm_table *t = NULL;
    rapidglm_bulk_options bulk;
    rapidglm_fit_options fit;
    size_t length = 0, j, called = 0;
    const double *padj, *sf;

    build();
    counts.rows = SAMPLES;
    counts.cols = GENES;
    counts.column_ptr = column_ptr;
    counts.row_index = row_index;
    counts.values = values;
    CHECK(rapidglm_c_api_version() == RAPIDGLM_C_API_VERSION);
    CHECK(strlen(rapidglm_version()) > 0);

    rapidglm_fit_options_init(&fit);
    CHECK(rapidglm_fit_batch(&x, &counts, &fit, 2, &t) == RAPIDGLM_OK);
    CHECK(rapidglm_table_rows(t) == GENES);
    CHECK(rapidglm_table_numeric(t, "beta1") != NULL);
    CHECK(strcmp(rapidglm_table_text(t, "status", 0), "converged") == 0);
    rapidglm_table_free(t);

    rapidglm_bulk_options_init(&bulk);
    bulk.lfc_shrinkage = RAPIDGLM_SHRINK_NORMAL;
    CHECK(rapidglm_bulk_de(&x, &counts, &bulk, &t) == RAPIDGLM_OK);
    CHECK(rapidglm_table_rows(t) == GENES);
    CHECK(rapidglm_table_columns(t) == 17);
    padj = rapidglm_table_numeric(t, "padj");
    for (j = 0; j < GENES; ++j)
        called += !isnan(padj[j]) && padj[j] < 0.1;
    CHECK(called >= 4);
    sf = rapidglm_table_vector(t, "size_factors", &length);
    CHECK(sf != NULL && length == SAMPLES);
    CHECK(rapidglm_table_scalar(t, "prior_variance") >= 0.25);
    CHECK(isnan(rapidglm_table_scalar(t, "no_such_value")));
    rapidglm_table_free(t);

    /* New 1.4 entry points are callable by a C99 client, including failure. */
    {
        rapidglm_ql_options ql;
        rapidglm_ql_options_init(&ql);
        CHECK(rapidglm_edger_ql(&x, &counts, &ql, &t) == RAPIDGLM_OK);
        CHECK(rapidglm_table_numeric(t, "F") != NULL);
        CHECK(isfinite(rapidglm_table_scalar(t, "dispersion")));
        rapidglm_table_free(t);
        values[0] += 0.5;
        t = NULL;
        CHECK(rapidglm_edger_ql(&x, &counts, &ql, &t) != RAPIDGLM_OK);
        CHECK(t == NULL);
        values[0] -= 0.5;
    }
    {
        double dense[SAMPLES * GENES], factors[SAMPLES], disp[GENES];
        double beta[GENES * 2], mu[SAMPLES * GENES], hat[SAMPLES * GENES];
        int converged[GENES];
        rapidglm_dense y = {SAMPLES, GENES, dense};
        for (j = 0; j < SAMPLES; ++j)
            factors[j] = 1;
        for (j = 0; j < GENES; ++j) {
            size_t i;
            disp[j] = .1;
            for (i = 0; i < SAMPLES; ++i)
                dense[i * GENES + j] = values[column_ptr[j] + i];
        }
        CHECK(rapidglm_nb_floored_fits(&x, &y, factors, disp, .5, 1e-8, 250, -30, 30, 1, beta, mu,
                                       hat, converged) == RAPIDGLM_OK);
        CHECK(converged[0] && isfinite(beta[1]) && hat[0] >= 0);
        {
            rapidglm_dense means = {SAMPLES, GENES, mu};
            CHECK(rapidglm_nb_dispersions(&x, &y, &means, disp, 1e-8, 10, 0, 1, 1, 1, disp,
                                          converged) == RAPIDGLM_OK);
            CHECK(disp[0] > 0 && converged[0]);
        }
    }

    /* A joint Wald test is refused with a message. */
    {
        const double rows[] = {1, 0, 0, 1};
        rapidglm_bulk_options_init(&bulk);
        bulk.contrasts = rows;
        bulk.contrast_rows = 2;
        t = NULL;
        CHECK(rapidglm_bulk_de(&x, &counts, &bulk, &t) != RAPIDGLM_OK);
        CHECK(t == NULL && strlen(rapidglm_last_error()) > 0);
    }
    if (failures)
        fprintf(stderr, "%d C interface checks failed\n", failures);
    else
        printf("C interface checks passed\n");
    return failures != 0;
}
