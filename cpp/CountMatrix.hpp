#pragma once

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
