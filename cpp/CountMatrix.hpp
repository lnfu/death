#pragma once

/*
 * Copyright (c) 2025, Enfu Liao <efliao@cs.nycu.edu.tw>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <filesystem>

class CountMatrix
{
public:
    // default constructor
    CountMatrix() = delete;
    CountMatrix(size_t n_gene, size_t n_sample);
    CountMatrix(
        size_t n_gene, size_t n_sample,
        char **gene_names, char **sample_names,
        double *data);

    CountMatrix(CountMatrix const &);            // copy constructor.
    CountMatrix &operator=(CountMatrix const &); // copy assignment operator.

    CountMatrix(CountMatrix &&) noexcept;            // move constructor.
    CountMatrix &operator=(CountMatrix &&) noexcept; // move assignment operator.

    ~CountMatrix();

    void swap(CountMatrix &other) noexcept;

    double at(size_t gene_idx, size_t sample_idx) const;
    double &at(size_t gene_idx, size_t sample_idx);

    size_t n_gene() const;
    size_t n_sample() const;

    static CountMatrix from_file(const std::filesystem::path &filepath);
    static CountMatrix from_file(const std::string &filepath);
    static CountMatrix from_file(const char *filepath);

private:
    size_t n_gene_;
    size_t n_sample_;
    char **gene_names_;
    char **sample_names_;
    double *data_;
};
