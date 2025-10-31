/* ----------------------------------------------------------------------
 * Project:  TinyTTA Engine
 *
 * Reference Paper:
 *  TinyTTA: Efficient Test-time Adaptation via Early-exit Ensembles on Edge Devices,
 *  Neural Information Processing Systems (NeurIPS) 2024
 *
 * Contact Authors:
 *  Young D. Kwon: ydk21@cam.ac.uk
 *  Hong Jia: hong.jia@unimelb.edu.au
 *  Alessio Orsino: aorsino@dimes.unical.it
 *  Ting Dang: ting.dang@unimelb.edu.au
 *  Domenico Talia: talia@dimes.unical.it
 *  Cecilia Mascolo: cm542@cam.ac.uk
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "utils.h"

int argMax(const int rNumClasses, TfLiteTensor* rOutput_ptr) {
    int idx_largest_num = 0;
    int8_t largest_num = rOutput_ptr->data.int8[0];
    for (int i = 1; i < rNumClasses; i++) {
        if (largest_num < rOutput_ptr->data.int8[i]) {
            largest_num = rOutput_ptr->data.int8[i];
            idx_largest_num = i;
        }
    }
    return idx_largest_num;
}

void printDouble(double v, int decimalDigits = 3) {
    int i = 1;
    int intPart, fractPart;
    for (; decimalDigits != 0; i *= 10, decimalDigits--)
        ;
    intPart = (int)v;
    fractPart = (int)((v - (double)(int)v) * i);
    if (fractPart < 0) fractPart *= -1;
    printf("%i.%i", intPart, fractPart);
}

void printFloat(float v, int decimalDigits = 3) {
    int i = 1;
    int intPart, fractPart;
    for (; decimalDigits != 0; i *= 10, decimalDigits--)
        ;
    intPart = (int)v;
    fractPart = (int)((v - (float)(int)v) * i);
    if (fractPart < 0) fractPart *= -1;
    printf("%i.%i", intPart, fractPart);
}
