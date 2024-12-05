#include "conv.h"
#include <iostream>
#include <ctime>
#include <cstring>

using namespace std;

#define C_in 1
#define C_out 16
#define Nin 16
#define Nout 16

data_t in[C_in][Nin][Nin];
data_t weight[C_out][C_in][K][K];
data_t bias[C_out];
data_t out[C_out][Nout][Nout];
data_t pool_out[C_out][Nout / 2][Nout / 2];
data_t exp_out[C_out][Nout][Nout];
data_t exp_pool_out[C_out][Nout / 2][Nout / 2];

int main() {
    // Initialize input data
    for (int i = 0; i < Nin; i++)
        for (int j = 0; j < Nin; j++)
            for (int n = 0; n < C_in; n++) {
                in[n][i][j] = (data_t)(((i * i * 3 - 6 * i + 7 * j * j - 9 * j + n * n * n - 13 * n + 19) % 1024 - 512) / 750.0);
            }

    // Initialize weight data
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            for (int m = 0; m < C_out; m++)
                for (int n = 0; n < C_in; n++) {
                    weight[m][n][i][j] = (data_t)(((m * m + 7 * m - 6 * n * n - 13 * n + i * i * i - 21 * i + 13 * j * j - 43) % 1024 - 512) / 600.0);
                }

    // Initialize bias data
    for (int m = 0; m < C_out; m++) {
        bias[m] = (data_t)(((m * m * m - 17 * m * m + 13 * m - 23) % 128 - 64) / 100.0);
    }

    // Compute expected output for verification
    int r, c, i, j, m, n;
    for (r = 0; r < Nout; r++)
        for (c = 0; c < Nout; c++)
            for (m = 0; m < C_out; m++) {
                exp_out[m][r][c] = bias[m];
                for (n = 0; n < C_in; n++)
                    for (i = 0; i < K; i++)
                        for (j = 0; j < K; j++) {
                            if ((r - pad + i) >= 0 && (r - pad + i) < Nin && (c - pad + j) >= 0 && (c - pad + j) < Nin)
                                exp_out[m][r][c] += (in[n][r - pad + i][c - pad + j] * weight[m][n][i][j]);
                        }
                exp_out[m][r][c] = (exp_out[m][r][c] > (data_t)0) ? exp_out[m][r][c] : (data_t)0; // ReLU activation
            }

    // Compute expected pooled output for verification
    for (r = 0; r < Nout / 2; r++)
        for (c = 0; c < Nout / 2; c++)
            for (m = 0; m < C_out; m++) {
                exp_pool_out[m][r][c] = (data_t)0;
                for (int kx = 0; kx < 2; kx++)
                    for (int ky = 0; ky < 2; ky++)
                        if (exp_out[m][2 * r + kx][2 * c + ky] > exp_pool_out[m][r][c])
                            exp_pool_out[m][r][c] = exp_out[m][2 * r + kx][2 * c + ky];
            }

    // Run the convolution function
    int pool = false;
    if (pool) {
        conv((volatile data_t*)in, (volatile data_t*)weight, (volatile data_t*)bias, (volatile data_t*)pool_out, C_in, C_out, Nin, 1);
    }
    else {
        conv((volatile data_t*)in, (volatile data_t*)weight, (volatile data_t*)bias, (volatile data_t*)out, C_in, C_out, Nin, 0);
    }

    // Verify output
    if (pool) {
        cout << "Conv + Pool Test" << endl;
        for (int m = 0; m < C_out; m++)
            for (int r = 0; r < Nout / 2; r++)
                for (int c = 0; c < Nout / 2; c++) {
                    if (exp_pool_out[m][r][c] != pool_out[m][r][c])
                        return -1;
                    cout << exp_pool_out[m][r][c] << "," << pool_out[m][r][c] << endl;
                }
    }
    else {
        cout << "Conv Only Test" << endl;
        for (int m = 0; m < C_out; m++)
            for (int r = 0; r < Nout; r++)
                for (int c = 0; c < Nout; c++) {
                    if (exp_out[m][r][c] != out[m][r][c]) {
                        return -1;
                    }
                    cout << exp_out[m][r][c] << "," << out[m][r][c] << endl;
                }
    }

    cout << "Test Pass" << endl;
    return 0;
}
