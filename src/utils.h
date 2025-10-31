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

#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <cstdio>
#include <random>

#include "tensorflow/lite/c/common.h"

int argMax(const int rNumClasses, TfLiteTensor* rOutput_ptr);
void printDouble(double v, int decimalDigits);
void printFloat(float v, int decimalDigits);

#endif  // UTILS_H
