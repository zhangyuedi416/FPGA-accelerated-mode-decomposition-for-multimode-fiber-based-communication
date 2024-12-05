#include<ap_fixed.h>
#include<ap_int.h>
#include<string.h>
#define Tr 16
#define Tc 16
#define Tn 8
#define Tm 16
#define K 3  
#define pad ((K-1)/2)
#define MAX_LEN 100

typedef ap_fixed<16,4,AP_RND,AP_SAT,0> data_t;

void conv(volatile data_t* in,volatile data_t* weight,volatile data_t* bias,volatile data_t* out,ap_int<9> N,ap_int<9> M,ap_int<9> SIZE,ap_int<9> pool);
