#ifndef VALIDATION_H
#define VALIDATION_H

struct ValidationResult {
    double max_abs_error;
    double max_rel_error;
};

ValidationResult validate_results(const float* ref, const float* test, int N);

#endif