#include "conv.h"

// Generic register function template
template<class T>
T Reg(T in){
#pragma HLS INTERFACE register port=return
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return in;
}

// Function to load input tile from memory into buffer
void load_input(data_t fm_in_buff[Tn][Tr+K-1][Tc+K-1], volatile data_t* fm_in,
                ap_int<9> n, ap_int<9> fm_row, ap_int<9> fm_col, ap_int<9> fm_size){
    ap_int<9> nn, rr, cc;
    for(nn = 0; nn < Tn; nn++)
        for(rr = 0; rr < Tr + K - 1; rr++)
            for(cc = 0; cc < Tc + K - 1; cc++){
                #pragma HLS PIPELINE II=1
                ap_int<9> r = Reg(rr + fm_row - pad);
                ap_int<9> c = Reg(cc + fm_col - pad);
                ap_int<18> size = Reg(fm_size * fm_size);
                ap_int<18> offset = Reg(r * fm_size + c) + Reg((n + nn) * size);
                if(r >= 0 && r < fm_size && c >= 0 && c < fm_size)
                    fm_in_buff[nn][rr][cc] = *(fm_in + offset);
                else
                    fm_in_buff[nn][rr][cc] = (data_t)0;
            }
}

// Function to load weights from memory into buffer
void load_weight(data_t wt_buff[Tm][Tn][K][K], volatile data_t* weight_in,
                 ap_int<9> n, ap_int<9> m, ap_int<9> channel_in, ap_int<9> channel_out){
    ap_int<9> mm, nn, rr, cc;
    for(mm = 0; mm < Tm; mm++)
        for(nn = 0; nn < Tn; nn++){
            if((n + nn) < channel_in && (m + mm) < channel_out){
                ap_int<18> offset = (m + mm) * channel_in * K * K + (n + nn) * K * K;
                memcpy((data_t*)wt_buff[mm][nn], (const data_t*)(weight_in + offset), K * K * sizeof(data_t));
            }
            else{
                for(rr = 0; rr < K; rr++)
                    for(cc = 0; cc < K; cc++)
                        #pragma HLS PIPELINE
                        wt_buff[mm][nn][rr][cc] = (data_t)0;
            }
        }
}

// Function to perform convolution computation
void compute(data_t fm_in_buff[Tn][Tr+K-1][Tc+K-1], data_t fm_out_buff[Tm][Tr][Tc], data_t wt_buff[Tm][Tn][K][K]){
#pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=fm_out_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=1
    int kx, ky, rr, cc, nn, mm;
    for(kx = 0; kx < K; kx++)
        for(ky = 0; ky < K; ky++)
            for(rr = 0; rr < Tr; rr++)
                for(cc = 0; cc < Tc; cc++)
                    #pragma HLS PIPELINE II=1
                    for(mm = 0; mm < Tm; mm++)
                        for(nn = 0; nn < Tn; nn++){
                            data_t mult = fm_in_buff[nn][rr + kx][cc + ky] * wt_buff[mm][nn][kx][ky];
                            data_t psum = mult + fm_out_buff[mm][rr][cc];
                            fm_out_buff[mm][rr][cc] = psum;
                        }
}

// Function to load bias into output buffer
void load_bias(data_t fm_out_buff[Tm][Tr][Tc], data_t bias_buff[MAX_LEN], ap_int<9> m){
    ap_int<9> rr, cc, mm;
    for(rr = 0; rr < Tr; rr++)
        for(cc = 0; cc < Tc; cc++)
            for(mm = 0; mm < Tm; mm++)
                #pragma HLS PIPELINE II=1
                fm_out_buff[mm][rr][cc] = bias_buff[m + mm];
}

// Function to store output from buffer to memory
void store_output(data_t fm_out_buff[Tm][Tr][Tc], volatile data_t* fm_out, ap_int<9> fm_row,
                  ap_int<9> fm_col, ap_int<9> m, ap_int<9> fm_size, ap_int<9> pool, ap_int<9> channel_out){
    ap_int<9> mm, rr, cc;
    if(pool == 0){
        for(mm = 0; mm < Tm; mm++)
            for(rr = 0; rr < Tr; rr++)
                for(cc = 0; cc < Tc; cc++){
                    #pragma HLS PIPELINE
                    ap_int<9> r = Reg(fm_row + rr);
                    ap_int<9> c = Reg(cc + fm_col);
                    ap_int<18> size = Reg(fm_size * fm_size);
                    ap_int<18> offset = Reg(Reg(r * fm_size) + c + Reg((mm + m) * size));
                    if(r < fm_size && c < fm_size && (m + mm) < channel_out)
                        *(fm_out + offset) = fm_out_buff[mm][rr][cc] > (data_t)0 ? fm_out_buff[mm][rr][cc] : (data_t)0; // ReLU activation
                }
    }
    else{
        ap_int<9> kx, ky;
        for(rr = 0; rr < Tr / 2; rr++)
            for(cc = 0; cc < Tc / 2; cc++)
                for(mm = 0; mm < Tm; mm++){
                    #pragma HLS PIPELINE
                    data_t tmp = (data_t)0; // ReLU equivalent
                    for(kx = 0; kx < 2; kx++)
                        for(ky = 0; ky < 2; ky++)
                            if(fm_out_buff[mm][2 * rr + kx][2 * cc + ky] > tmp)
                                tmp = fm_out_buff[mm][2 * rr + kx][2 * cc + ky]; // 2x2 max pooling
                    ap_int<9> r = Reg(fm_row / 2 + rr);
                    ap_int<9> c = Reg(fm_col / 2 + cc);
                    ap_int<18> size = Reg(fm_size * fm_size / 4);
                    ap_int<18> offset = Reg(r * fm_size / 2 + c) + Reg((m + mm) * size);
                    if(r < fm_size / 2 && c < fm_size / 2 && (m + mm) < channel_out)
                        *(fm_out + offset) = tmp;
                }
    }
}

// Function to handle computation of output tiles
void compute_output(data_t fm_in_buff1[Tn][Tr+K-1][Tc+K-1], data_t fm_in_buff2[Tn][Tr+K-1][Tc+K-1],
                    data_t wt_buff1[Tm][Tn][K][K], data_t wt_buff2[Tm][Tn][K][K], data_t fm_out_buff[Tm][Tr][Tc],
                    data_t bias_buff[MAX_LEN], volatile data_t* fm_in, volatile data_t* weight_in,
                    ap_int<9> m, ap_int<9> fm_size, ap_int<9> channel_in, ap_int<9> channel_out, ap_int<9> fm_row, ap_int<9> fm_col){
    ap_int<9> ti = 0;
    ap_int<1> pingpong = 0;

    // Initialize input and weights for computation
    load_bias(fm_out_buff, bias_buff, m);
    load_input(fm_in_buff1, fm_in, ti, fm_row, fm_col, fm_size);
    load_weight(wt_buff1, weight_in, ti, m, channel_in, channel_out);

    for(ti = Tn; ti < channel_in; ti += Tn){
        if(pingpong == 0){
            load_input(fm_in_buff2, fm_in, ti, fm_row, fm_col, fm_size);
            load_weight(wt_buff2, weight_in, ti, m, channel_in, channel_out);
            compute(fm_in_buff1, fm_out_buff, wt_buff1);
            pingpong = 1;
        }
        else{
            load_input(fm_in_buff1, fm_in, ti, fm_row, fm_col, fm_size);
            load_weight(wt_buff1, weight_in, ti, m, channel_in, channel_out);
            compute(fm_in_buff2, fm_out_buff, wt_buff2);
            pingpong = 0;
        }
    }
    if(pingpong == 0)
        compute(fm_in_buff1, fm_out_buff, wt_buff1);
    else
        compute(fm_in_buff2, fm_out_buff, wt_buff2);
}

// Main convolution function
void conv(volatile data_t* in, volatile data_t* weight, volatile data_t* bias, volatile data_t* out, ap_int<9> N, ap_int<9> M, ap_int<9> SIZE, ap_int<9> pool){
#pragma HLS INTERFACE m_axi depth=16*16*10 port=in offset=slave bundle=IN
#pragma HLS INTERFACE m_axi depth=16*16*10 port=weight offset=slave bundle=W
#pragma HLS INTERFACE m_axi depth=32 port=bias offset=slave bundle=B
#pragma HLS INTERFACE m_axi depth=32*32*9 port=out offset=slave bundle=OUT
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite port=N bundle=CTRL
#pragma HLS INTERFACE s_axilite port=M bundle=CTRL
#pragma HLS INTERFACE s_axilite port=SIZE bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pool bundle=CTRL

    ap_int<9> row, col, to;
    data_t bias_buff[MAX_LEN];
    data_t fm_in1[Tn][Tr+K-1][Tc+K-1];
#pragma HLS ARRAY_PARTITION variable=fm_in1 complete dim=1
    data_t fm_in2[Tn][Tr+K-1][Tc+K-1];
#pragma HLS ARRAY_PARTITION variable=fm_in2 complete dim=1
    data_t wt1[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=wt1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt1 complete dim=1
    data_t wt2[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=wt2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt2 complete dim=1
    data_t fm_out1[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=1
    data_t fm_out2[Tm][Tr][Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=1

    ap_int<1> pingpong;
    memcpy((data_t*)bias_buff, (const data_t*)bias, sizeof(data_t) * M);

    for(row = 0; row < SIZE; row += Tr)
        for(col = 0; col < SIZE; col += Tc){
            to = 0;
            pingpong = 0;
            compute_output(fm_in1, fm_in2, wt1, wt2, fm_out1, bias_buff, in, weight, to, SIZE, N, M, row, col);
            for(to = Tm; to < M; to += Tm){
                if(pingpong == 0){
                    compute_output(fm_in1, fm_in2, wt1, wt2, fm_out2, bias_buff, in, weight, to, SIZE, N, M, row, col);
                    store_output(fm_out1, out, row, col, to - Tm, SIZE, pool, M);
                    pingpong = 1;
                }
                else{
                    compute_output(fm_in1, fm_in2, wt1, wt2, fm_out1, bias_buff, in, weight, to, SIZE, N, M, row, col);
                    store_output(fm_out2, out, row, col, to - Tm, SIZE, pool, M);
                    pingpong = 0;
                }
            }
            if(pingpong == 0)
                store_output(fm_out1, out, row, col, to - Tm, SIZE, pool, M);
            else
                store_output(fm_out2, out, row, col, to - Tm, SIZE, pool, M);
        }
}
