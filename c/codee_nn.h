int affine_s (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L);
int affine_d (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L);
int affine_c (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L);
int affine_z (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L);

int affine_cblas_s (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L);
int affine_cblas_d (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L);
int affine_cblas_c (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L);
int affine_cblas_z (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L);

int asinh_s (float *Y, const float *X, const size_t N);
int asinh_d (double *Y, const double *X, const size_t N);
int asinh_c (float *Y, const float *X, const size_t N);
int asinh_z (double *Y, const double *X, const size_t N);
int asinh_inplace_s (float *X, const size_t N);
int asinh_inplace_d (double *X, const size_t N);
int asinh_inplace_c (float *X, const size_t N);
int asinh_inplace_z (double *X, const size_t N);

int atan_s (float *Y, const float *X, const size_t N);
int atan_d (double *Y, const double *X, const size_t N);
int atan_c (float *Y, const float *X, const size_t N);
int atan_z (double *Y, const double *X, const size_t N);
int atan_inplace_s (float *X, const size_t N);
int atan_inplace_d (double *X, const size_t N);
int atan_inplace_c (float *X, const size_t N);
int atan_inplace_z (double *X, const size_t N);

int avgpool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int avgpool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int avgpool1_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int avgpool1_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);

int avgpool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int avgpool1d_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int avgpool1d_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int avgpool1d_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);

int betamax_s (float *Y, const float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const float base);
int betamax_d (double *Y, const double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const double base);
int betamax_inplace_s (float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const float base);
int betamax_inplace_d (double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const double base);

int bilinear_s (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_d (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_c (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_z (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);

int bilinear_cblas_s (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_cblas_d (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_cblas_c (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_cblas_z (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);

int celu_s (float *Y, const float *X, const size_t N, const float alpha);
int celu_d (double *Y, const double *X, const size_t N, const double alpha);
int celu_inplace_s (float *X, const size_t N, const float alpha);
int celu_inplace_d (double *X, const size_t N, const double alpha);

int cmp_ascend_s (const void *a, const void *b);
int cmp_ascend_d (const void *a, const void *b);
int cmp_ascend_c (const void *a, const void *b);
int cmp_ascend_z (const void *a, const void *b);

int cmp_descend_s (const void *a, const void *b);
int cmp_descend_d (const void *a, const void *b);
int cmp_descend_c (const void *a, const void *b);
int cmp_descend_z (const void *a, const void *b);

int conv1_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);

int conv1_cblas_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_cblas_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_cblas_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_cblas_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);

int conv1d_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);

int conv1d_cblas_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_cblas_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_cblas_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_cblas_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);

int conv1d_torch_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_torch_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);

int conv1g_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1g_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1g_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1g_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);

int conv1g_torch_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1g_torch_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);

int conv1_torch_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_torch_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);

int elman_s (float *Y, const float *X, const float *U, float *H, const float *W, const float *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int elman_d (double *Y, const double *X, const double *U, double *H, const double *W, const double *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int elu_s (float *Y, const float *X, const size_t N, const float alpha);
int elu_d (double *Y, const double *X, const size_t N, const double alpha);
int elu_inplace_s (float *X, const size_t N, const float alpha);
int elu_inplace_d (double *X, const size_t N, const double alpha);

int erf_s (float *Y, const float *X, const size_t N);
int erf_d (double *Y, const double *X, const size_t N);
int erf_inplace_s (float *X, const size_t N);
int erf_inplace_d (double *X, const size_t N);

int fir_s (float *Y, const float *X, const float *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim);
int fir_d (double *Y, const double *X, const double *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim);
int fir_c (float *Y, const float *X, const float *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim);
int fir_z (double *Y, const double *X, const double *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim);

int fukushima2_s (float *Y, const float *Xe, const float *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima2_d (double *Y, const double *Xe, const double *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima2_inplace_s (float *Xe, const float *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima2_inplace_d (double *Xe, const double *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int fukushima_s (float *Y, const float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_d (double *Y, const double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_inplace_s (float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_inplace_d (double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int fukushima_s (float *Y, const float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_d (double *Y, const double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_inplace_s (float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_inplace_d (double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gelu_s (float *Y, const float *X, const size_t N);
int gelu_d (double *Y, const double *X, const size_t N);
int gelu_inplace_s (float *X, const size_t N);
int gelu_inplace_d (double *X, const size_t N);

int gelu_new_s (float *Y, const float *X, const size_t N);
int gelu_new_d (double *Y, const double *X, const size_t N);
int gelu_new_inplace_s (float *X, const size_t N);
int gelu_new_inplace_d (double *X, const size_t N);

int grossberg2_s (float *Y, const float *Xe, const float *Xi, const float *tau, const float *alpha, const float *betae, const float *betai, const float *gammae, const float *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg2_d (double *Y, const double *Xe, const double *Xi, const double *tau, const double *alpha, const double *betae, const double *betai, const double *gammae, const double *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int grossberg2_inplace_s (float *Xe, const float *Xi, const float *tau, const float *alpha, const float *betae, const float *betai, const float *gammae, const float *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg2_inplace_d (double *Xe, const double *Xi, const double *tau, const double *alpha, const double *betae, const double *betai, const double *gammae, const double *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);

int grossberg_s (float *Y, const float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg_d (double *Y, const double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int grossberg_c (float *Y, const float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg_z (double *Y, const double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int grossberg_inplace_s (float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg_inplace_d (double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int grossberg_inplace_c (float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg_inplace_z (double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);

int gru3_s (float *Y, const float *X, const float *Xr, const float *Xz, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru3_d (double *Y, const double *X, const double *Xr, const double *Xz, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru3_inplace_s (float *X, const float *Xr, const float *Xz, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru3_inplace_d (double *X, const double *Xr, const double *Xz, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gru_s (float *Y, const float *X, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_d (double *Y, const double *X, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_inplace_s (float *X, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_inplace_d (double *X, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gru_min2_s (float *Y, const float *X, const float *Xf, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min2_d (double *Y, const double *X, const double *Xf, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min2_inplace_s (float *X, const float *Xf, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min2_inplace_d (double *X, const double *Xf, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gru_min_s (float *Y, const float *X, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min_d (double *Y, const double *X, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min_inplace_s (float *X, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min_inplace_d (double *X, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gudermann_s (float *Y, const float *X, const size_t N);
int gudermann_d (double *Y, const double *X, const size_t N);
int gudermann_c (float *Y, const float *X, const size_t N);
int gudermann_z (double *Y, const double *X, const size_t N);
int gudermann_inplace_s (float *X, const size_t N);
int gudermann_inplace_d (double *X, const size_t N);
int gudermann_inplace_c (float *X, const size_t N);
int gudermann_inplace_z (double *X, const size_t N);

int hardshrink_s (float *Y, const float *X, const size_t N, const float lambda);
int hardshrink_d (double *Y, const double *X, const size_t N, const double lambda);
int hardshrink_inplace_s (float *X, const size_t N, const float lambda);
int hardshrink_inplace_d (double *X, const size_t N, const double lambda);

int hardsigmoid_s (float *Y, const float *X, const size_t N);
int hardsigmoid_d (double *Y, const double *X, const size_t N);
int hardsigmoid_inplace_s (float *X, const size_t N);
int hardsigmoid_inplace_d (double *X, const size_t N);

int hardswish_s (float *Y, const float *X, const size_t N);
int hardswish_d (double *Y, const double *X, const size_t N);
int hardswish_inplace_s (float *X, const size_t N);
int hardswish_inplace_d (double *X, const size_t N);

int hardtanh_s (float *Y, const float *X, const size_t N, const float a, const float b);
int hardtanh_d (double *Y, const double *X, const size_t N, const double a, const double b);
int hardtanh_inplace_s (float *X, const size_t N, const float a, const float b);
int hardtanh_inplace_d (double *X, const size_t N, const double a, const double b);

int hopfield_s (float *Y, const float *X, const float *tau, const float *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int hopfield_d (double *Y, const double *X, const double *tau, const double *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int hopfield_c (float *Y, const float *X, const float *tau, const float *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int hopfield_z (double *Y, const double *X, const double *tau, const double *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int hopfield_inplace_s (float *X, const float *tau, const float *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int hopfield_inplace_d (double *X, const double *tau, const double *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int hopfield_inplace_c (float *X, const float *tau, const float *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int hopfield_inplace_z (double *X, const double *tau, const double *alpha, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);

int identity_s (float *Y, const float *X, const size_t N);
int identity_d (double *Y, const double *X, const size_t N);
int identity_c (float *Y, const float *X, const size_t N);
int identity_z (double *Y, const double *X, const size_t N);

int iir_s (float *Y, const float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_d (double *Y, const double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_c (float *Y, const float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_z (double *Y, const double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_inplace_s (float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_inplace_d (double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_inplace_c (float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_inplace_z (double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);

int integrate_s (float *Y, const float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_d (double *Y, const double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int integrate_c (float *Y, const float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_z (double *Y, const double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int integrate_inplace_s (float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_inplace_d (double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int integrate_inplace_c (float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_inplace_z (double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);

int isrlu_s (float *Y, const float *X, const size_t N, const float alpha);
int isrlu_d (double *Y, const double *X, const size_t N, const double alpha);
int isrlu_inplace_s (float *X, const size_t N, const float alpha);
int isrlu_inplace_d (double *X, const size_t N, const double alpha);

int isru_s (float *Y, const float *X, const size_t N, const float alpha);
int isru_d (double *Y, const double *X, const size_t N, const double alpha);
int isru_inplace_s (float *X, const size_t N, const float alpha);
int isru_inplace_d (double *X, const size_t N, const double alpha);

int jordan_s (float *Y, const float *X, const float *U, float *Y1, const float *W, const float *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int jordan_d (double *Y, const double *X, const double *U, double *Y1, const double *W, const double *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int leaky_relu_s (float *Y, const float *X, const size_t N, const float alpha);
int leaky_relu_d (double *Y, const double *X, const size_t N, const double alpha);
int leaky_relu_inplace_s (float *X, const size_t N, const float alpha);
int leaky_relu_inplace_d (double *X, const size_t N, const double alpha);

int linear_s (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_d (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);
int linear_c (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_z (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);

int linear_cblas_s (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_cblas_d (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);
int linear_cblas_c (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_cblas_z (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);

int logistic_s (float *Y, const float *X, const size_t N, const float alpha);
int logistic_d (double *Y, const double *X, const size_t N, const double alpha);
int logistic_inplace_s (float *X, const size_t N, const float alpha);
int logistic_inplace_d (double *X, const size_t N, const double alpha);

int logsigmoid_s (float *Y, const float *X, const size_t N);
int logsigmoid_d (double *Y, const double *X, const size_t N);
int logsigmoid_inplace_s (float *X, const size_t N);
int logsigmoid_inplace_d (double *X, const size_t N);

int log_softmax_s (float *Y, const float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int log_softmax_d (double *Y, const double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int log_softmax_inplace_s (float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int log_softmax_inplace_d (double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);

int lppool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode, const float pw);
int lppool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode, const double pw);
int lppool1_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode, const float pw);
int lppool1_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode, const double pw);

int lppool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode, const float pw);
int lppool1d_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode, const double pw);
int lppool1d_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode, const float pw);
int lppool1d_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode, const double pw);

int lstm4_s (float *Y, const float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm4_d (double *Y, const double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm4_inplace_s (float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm4_inplace_d (double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int lstm_s (float *Y, const float *X, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_d (double *Y, const double *X, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_inplace_s (float *X, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_inplace_d (double *X, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int lstm_peephole4_s (float *Y, const float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_peephole4_d (double *Y, const double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_peephole4_inplace_s (float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_peephole4_inplace_d (double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int lstm_peephole_s (float *Y, const float *X, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_peephole_d (double *Y, const double *X, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_peephole_inplace_s (float *X, const float *Ui, const float *Uf, const float *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int lstm_peephole_inplace_d (double *X, const double *Ui, const double *Uf, const double *Uo, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int maxout_s (float *Y, const float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const size_t M);
int maxout_d (double *Y, const double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const size_t M);

int maxpool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);
int maxpool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);
int maxpool1_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);
int maxpool1_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);

int maxpool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int maxpool1d_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int maxpool1d_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int maxpool1d_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);

int mish_s (float *Y, const float *X, const size_t N);
int mish_d (double *Y, const double *X, const size_t N);
int mish_inplace_s (float *X, const size_t N);
int mish_inplace_d (double *X, const size_t N);

int perceptron_s (float *Y, const float *X, const size_t N, const float thresh);
int perceptron_d (double *Y, const double *X, const size_t N, const double thresh);
int perceptron_inplace_s (float *X, const size_t N, const float thresh);
int perceptron_inplace_d (double *X, const size_t N, const double thresh);

int plu_s (float *Y, const float *X, const size_t N, float a, float c);
int plu_d (double *Y, const double *X, const size_t N, double a, double c);
int plu_inplace_s (float *X, const size_t N, float a, float c);
int plu_inplace_d (double *X, const size_t N, double a, double c);

int prelu_s (float *Y, const float *X, const size_t N, const float alpha);
int prelu_d (double *Y, const double *X, const size_t N, const double alpha);
int prelu_inplace_s (float *X, const size_t N, const float alpha);
int prelu_inplace_d (double *X, const size_t N, const double alpha);

int relu6_s (float *Y, const float *X, const size_t N);
int relu6_d (double *Y, const double *X, const size_t N);
int relu6_inplace_s (float *X, const size_t N);
int relu6_inplace_d (double *X, const size_t N);

int relu_s (float *Y, const float *X, const size_t N);
int relu_d (double *Y, const double *X, const size_t N);
int relu_inplace_s (float *X, const size_t N);
int relu_inplace_d (double *X, const size_t N);

int rrelu_s (float *Y, const float *X, const size_t N, const float lower, const float upper);
int rrelu_d (double *Y, const double *X, const size_t N, const double lower, const double upper);
int rrelu_inplace_s (float *X, const size_t N, const float lower, const float upper);
int rrelu_inplace_d (double *X, const size_t N, const double lower, const double upper);

int selu_s (float *Y, const float *X, const size_t N);
int selu_d (double *Y, const double *X, const size_t N);
int selu_inplace_s (float *X, const size_t N);
int selu_inplace_d (double *X, const size_t N);

int sigmoid_s (float *Y, const float *X, const size_t N);
int sigmoid_d (double *Y, const double *X, const size_t N);
int sigmoid_inplace_s (float *X, const size_t N);
int sigmoid_inplace_d (double *X, const size_t N);

int signum_s (float *Y, const float *X, const size_t N, const float thresh);
int signum_d (double *Y, const double *X, const size_t N, const double thresh);
int signum_inplace_s (float *X, const size_t N, const float thresh);
int signum_inplace_d (double *X, const size_t N, const double thresh);

int silu_s (float *Y, const float *X, const size_t N);
int silu_d (double *Y, const double *X, const size_t N);
int silu_inplace_s (float *X, const size_t N);
int silu_inplace_d (double *X, const size_t N);

int sin_s (float *Y, const float *X, const size_t N);
int sin_d (double *Y, const double *X, const size_t N);
int sin_c (float *Y, const float *X, const size_t N);
int sin_z (double *Y, const double *X, const size_t N);
int sin_inplace_s (float *X, const size_t N);
int sin_inplace_d (double *X, const size_t N);
int sin_inplace_c (float *X, const size_t N);
int sin_inplace_z (double *X, const size_t N);

int smoothstep_s (float *Y, const float *X, const size_t N, const int p);
int smoothstep_d (double *Y, const double *X, const size_t N, const int p);
int smoothstep_inplace_s (float *X, const size_t N, const int p);
int smoothstep_inplace_d (double *X, const size_t N, const int p);

int softclip_s (float *Y, const float *X, const size_t N, const float p);
int softclip_d (double *Y, const double *X, const size_t N, const double p);
int softclip_inplace_s (float *X, const size_t N, const float p);
int softclip_inplace_d (double *X, const size_t N, const double p);

int softmax_s (float *Y, const float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int softmax_d (double *Y, const double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int softmax_inplace_s (float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int softmax_inplace_d (double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);

int softmin_s (float *Y, const float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int softmin_d (double *Y, const double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int softmin_inplace_s (float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);
int softmin_inplace_d (double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim);

int softplus_s (float *Y, const float *X, const size_t N, const float beta, const float thresh);
int softplus_d (double *Y, const double *X, const size_t N, const double beta, const double thresh);
int softplus_inplace_s (float *X, const size_t N, const float beta, const float thresh);
int softplus_inplace_d (double *X, const size_t N, const double beta, const double thresh);

int softshrink_s (float *Y, const float *X, const size_t N, const float lambda);
int softshrink_d (double *Y, const double *X, const size_t N, const double lambda);
int softshrink_inplace_s (float *X, const size_t N, const float lambda);
int softshrink_inplace_d (double *X, const size_t N, const double lambda);

int softsign_s (float *Y, const float *X, const size_t N);
int softsign_d (double *Y, const double *X, const size_t N);
int softsign_inplace_s (float *X, const size_t N);
int softsign_inplace_d (double *X, const size_t N);

int sqnl_s (float *Y, const float *X, const size_t N);
int sqnl_d (double *Y, const double *X, const size_t N);
int sqnl_inplace_s (float *X, const size_t N);
int sqnl_inplace_d (double *X, const size_t N);

int step_s (float *Y, const float *X, const size_t N, const float thresh);
int step_d (double *Y, const double *X, const size_t N, const double thresh);
int step_inplace_s (float *X, const size_t N, const float thresh);
int step_inplace_d (double *X, const size_t N, const double thresh);

int swish_s (float *Y, const float *X, const size_t N, const float beta);
int swish_d (double *Y, const double *X, const size_t N, const double beta);
int swish_inplace_s (float *X, const size_t N, const float beta);
int swish_inplace_d (double *X, const size_t N, const double beta);

int tanh_s (float *Y, const float *X, const size_t N);
int tanh_d (double *Y, const double *X, const size_t N);
int tanh_c (float *Y, const float *X, const size_t N);
int tanh_z (double *Y, const double *X, const size_t N);
int tanh_inplace_s (float *X, const size_t N);
int tanh_inplace_d (double *X, const size_t N);
int tanh_inplace_c (float *X, const size_t N);
int tanh_inplace_z (double *X, const size_t N);

int tanhshrink_s (float *Y, const float *X, const size_t N);
int tanhshrink_d (double *Y, const double *X, const size_t N);
int tanhshrink_inplace_s (float *X, const size_t N);
int tanhshrink_inplace_d (double *X, const size_t N);

int threshold_s (float *Y, const float *X, const size_t N, const float thresh, const float val);
int threshold_d (double *Y, const double *X, const size_t N, const double thresh, const double val);
int threshold_inplace_s (float *X, const size_t N, const float thresh, const float val);
int threshold_inplace_d (double *X, const size_t N, const double thresh, const double val);

