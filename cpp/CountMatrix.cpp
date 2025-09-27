#include <vector>
#include <filesystem>
#include <iostream>

#include <cstddef>
#include <fstream>
#include <cstring>
#include <algorithm>

#include "CountMatrix.hpp"

CountMatrix::CountMatrix(size_t n_gene, size_t n_sample) : n_gene_(n_gene), n_sample_(n_sample)
{
    size_t nelement = n_gene * n_sample;
    data_ = new double[nelement];
    gene_names_ = new char *[n_gene];
    sample_names_ = new char *[n_sample];
}

CountMatrix::CountMatrix(
    size_t n_gene, size_t n_sample,
    char **gene_names, char **sample_names,
    double *data)
    : CountMatrix(n_gene, n_sample)
{
    if (!data)
    {
        throw std::runtime_error("Data pointer is null.");
    }
    if (!gene_names)
    {
        throw std::runtime_error("Gene names pointer is null.");
    }
    if (!sample_names)
    {
        throw std::runtime_error("Sample names pointer is null.");
    }

    size_t nelement = n_gene * n_sample;
    std::copy(data, data + nelement, data_);

    for (size_t i = 0; i < n_gene; ++i)
    {
        gene_names_[i] = new char[std::strlen(gene_names[i]) + 1];
        std::strcpy(gene_names_[i], gene_names[i]);
    }

    for (size_t i = 0; i < n_sample; ++i)
    {
        sample_names_[i] = new char[std::strlen(sample_names[i]) + 1];
        std::strcpy(sample_names_[i], sample_names[i]);
    }
}

CountMatrix::~CountMatrix()
{
    delete[] data_;

    if (gene_names_)
    {
        for (size_t i = 0; i < n_gene_; ++i)
            if (gene_names_[i])
                delete[] gene_names_[i];

        delete[] gene_names_;
    }

    if (sample_names_)
    {
        for (size_t i = 0; i < n_sample_; ++i)
            if (sample_names_[i])
                delete[] sample_names_[i];

        delete[] sample_names_;
    }
}

double CountMatrix::at(size_t gene_idx, size_t sample_idx) const
{
    if (gene_idx >= n_gene_ || sample_idx >= n_sample_)
        throw std::out_of_range("Index out of range in CountMatrix::at()");

    return data_[gene_idx * n_sample_ + sample_idx];
}

double &CountMatrix::at(size_t gene_idx, size_t sample_idx)
{
    if (gene_idx >= n_gene_ || sample_idx >= n_sample_)
        throw std::out_of_range("Index out of range in CountMatrix::at()");

    return data_[gene_idx * n_sample_ + sample_idx];
}

size_t CountMatrix::n_gene() const { return n_gene_; }
size_t CountMatrix::n_sample() const { return n_sample_; }

CountMatrix CountMatrix::from_file(const std::filesystem::path &filepath)
{
    if (filepath.extension() == ".txt")
    {
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filepath.string());
        }

        // Placeholder values; replace with actual parsing logic.
        size_t n_gene = 0;
        size_t n_sample = 0;
        std::vector<std::string> sample_names;
        std::vector<std::string> gene_names;
        std::vector<double> data;

        std::string line;

        // Read header line for sample names
        if (!std::getline(file, line))
        {
            throw std::runtime_error("File is empty: " + filepath.string());
        }
        std::string sample_name;
        std::istringstream iss(line);
        iss >> sample_name; // Skip the first column ("Geneid")
        while (iss >> sample_name)
        {
            sample_names.push_back(sample_name);
            n_sample++;
        }

        // Read gene names and data
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string gene_name;
            double count;

            iss >> gene_name;
            gene_names.push_back(gene_name);

            for (size_t i = 0; i < n_sample; ++i)
            {
                if (iss.eof())
                {
                    throw std::runtime_error("Unexpected end of line while reading data in file: " + filepath.string());
                }
                if (!(iss >> count))
                {
                    throw std::runtime_error("Failed to read count data in file: " + filepath.string());
                }
                data.push_back(count);
            }
            n_gene++;
        }

        file.close();
        if (data.size() != n_gene * n_sample)
        {
            std::cerr << "Data size: " << data.size() << ", n_gene: " << n_gene << ", n_sample: " << n_sample << "\n";
            throw std::runtime_error("Data size mismatch in file: " + filepath.string());
        }

        CountMatrix matrix(n_gene, n_sample);

        for (size_t i = 0; i < n_gene; ++i)
        {
            matrix.gene_names_[i] = new char[gene_names[i].size() + 1];
            std::strcpy(matrix.gene_names_[i], gene_names[i].c_str());
        }

        for (size_t i = 0; i < n_sample; ++i)
        {
            matrix.sample_names_[i] = new char[sample_names[i].size() + 1];
            std::strcpy(matrix.sample_names_[i], sample_names[i].c_str());
        }

        std::copy(data.begin(), data.end(), matrix.data_);

        return matrix;
    }
    else
    {
        throw std::runtime_error("Unsupported file format: " + filepath.string());
    }
}

CountMatrix CountMatrix::from_file(const std::string &filepath)
{
    return from_file(std::filesystem::path(filepath));
}

CountMatrix CountMatrix::from_file(const char *filepath)
{
    return from_file(std::filesystem::path(filepath));
}

CountMatrix::CountMatrix(CountMatrix const &other)
{
    n_gene_ = other.n_gene_;
    n_sample_ = other.n_sample_;

    size_t nelement = n_gene_ * n_sample_;
    data_ = new double[nelement];
    std::copy(other.data_, other.data_ + nelement, data_);

    gene_names_ = new char *[n_gene_];
    for (size_t i = 0; i < n_gene_; ++i)
    {
        gene_names_[i] = new char[std::strlen(other.gene_names_[i]) + 1];
        std::strcpy(gene_names_[i], other.gene_names_[i]);
    }

    sample_names_ = new char *[n_sample_];
    for (size_t i = 0; i < n_sample_; ++i)
    {
        sample_names_[i] = new char[std::strlen(other.sample_names_[i]) + 1];
        std::strcpy(sample_names_[i], other.sample_names_[i]);
    }
}

CountMatrix &CountMatrix::operator=(CountMatrix const &other)
{
    CountMatrix temp(other);
    swap(temp);
    return *this;
}

CountMatrix::CountMatrix(CountMatrix &&other) noexcept
    : n_gene_(other.n_gene_), n_sample_(other.n_sample_),
      gene_names_(other.gene_names_), sample_names_(other.sample_names_),
      data_(other.data_)
{
    other.n_gene_ = 0;
    other.n_sample_ = 0;
    other.data_ = nullptr;
    other.gene_names_ = nullptr;
    other.sample_names_ = nullptr;
}

CountMatrix &CountMatrix::operator=(CountMatrix &&other) noexcept
{
    CountMatrix temp(std::move(other));
    swap(temp);
    return *this;
}

void CountMatrix::swap(CountMatrix &other) noexcept
{
    std::swap(n_gene_, other.n_gene_);
    std::swap(n_sample_, other.n_sample_);
    std::swap(data_, other.data_);
    std::swap(gene_names_, other.gene_names_);
    std::swap(sample_names_, other.sample_names_);
}