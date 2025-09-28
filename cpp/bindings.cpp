#include <cstddef>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "CountMatrix.hpp"
namespace py = pybind11;

const char *init_doc = R"doc(
    CountMatrix(n_gene, n_sample)
    
    Initialize a count matrix with specified dimensions.
    
    Parameters
    ----------
    n_gene : int
        Number of genes (rows)
    n_sample : int
        Number of samples (columns)
)doc";

CountMatrix from_file(const py::object &filepath)
{
    // pathlib.Path
    if (py::hasattr(filepath, "__fspath__"))
    {
        std::string path_str = py::str(filepath.attr("__fspath__")());
        return CountMatrix::from_file(path_str);
    }
    // str || bytes
    else if (py::isinstance<py::str>(filepath) || py::isinstance<py::bytes>(filepath))
    {
        std::string path_str = filepath.cast<std::string>();
        return CountMatrix::from_file(path_str);
    }
    else
    {
        throw std::invalid_argument(
            "filepath must be a string or Path-like object");
    }
}

const char *from_file_doc = R"doc(
    Load count matrix from file
    
    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the count matrix file
        Supported formats: .txt, .tsv
    
    Returns
    -------
    CountMatrix
        Loaded count matrix
    
    Examples
    --------
    >>> # Using string
    >>> matrix = CountMatrix.from_file("GSE229869_cell.cycle.rnaseq.counts.txt")
    >>> 
    >>> # Using pathlib.Path
    >>> from pathlib import Path
    >>> matrix = CountMatrix.from_file(Path("GSE229869_cell.cycle.rnaseq.counts.txt"))
)doc";

std::string to_string(const CountMatrix &m)
{
    std::ostringstream oss;
    size_t n_genes = m.n_gene();
    size_t n_samples = m.n_sample();

    const size_t head_rows = 2, tail_rows = 2;
    const size_t head_cols = 3, tail_cols = 3;

    bool row_ellipsis = n_genes > (head_rows + tail_rows);
    bool col_ellipsis = n_samples > (head_cols + tail_cols);

    oss << "CountMatrix(" << n_genes << " genes x " << n_samples << " samples)\n\n";

    auto print_row = [&](size_t row)
    {
        size_t cols_head = std::min(head_cols, n_samples);
        for (size_t j = 0; j < cols_head; ++j)
        {
            if (j > 0)
                oss << ", ";
            oss << std::left << std::setw(6) << m.at(row, j);
        }

        if (col_ellipsis)
        {
            oss << ", ...";
            for (size_t j = n_samples - tail_cols; j < n_samples; ++j)
            {
                oss << ", " << std::setw(6) << m.at(row, j);
            }
        }
        else if (n_samples > head_cols)
        {
            for (size_t j = head_cols; j < n_samples; ++j)
            {
                oss << ", " << std::setw(6) << m.at(row, j);
            }
        }

        oss << "\n";
    };

    size_t show_head = std::min(head_rows, n_genes);
    for (size_t i = 0; i < show_head; ++i)
    {
        print_row(i);
    }

    if (row_ellipsis)
    {
        oss << "...\n";
        for (size_t i = n_genes - tail_rows; i < n_genes; ++i)
        {
            print_row(i);
        }
    }
    else if (n_genes > head_rows)
    {
        for (size_t i = head_rows; i < n_genes; ++i)
        {
            print_row(i);
        }
    }

    return oss.str();
}

std::string repr(const CountMatrix &m)
{
    return "<CountMatrix shape=(" +
           std::to_string(m.n_gene()) + ", " +
           std::to_string(m.n_sample()) + ")>";
}

double get_item(const CountMatrix &m, py::tuple index)
{
    if (index.size() != 2)
        throw std::invalid_argument("Need 2 indices");
    return m.at(index[0].cast<size_t>(),
                index[1].cast<size_t>());
}

void set_item(CountMatrix &m, py::tuple index, double value)
{
    if (index.size() != 2)
        throw std::invalid_argument("Need 2 indices");
    m.at(index[0].cast<size_t>(),
         index[1].cast<size_t>()) = value;
}

PYBIND11_MODULE(_libdeath, m)
{
    m.doc() = "High-performance gene expression count matrix for bioinformatics analysis.";

    py::class_<CountMatrix>(m, "CountMatrix")
        .def(py::init<size_t, size_t>(),
             py::arg("n_gene"), py::arg("n_sample"),
             init_doc)
        .def_property_readonly("n_gene", &CountMatrix::n_gene,
                               "Number of genes in the matrix")
        .def_property_readonly("n_sample", &CountMatrix::n_sample,
                               "Number of samples in the matrix")
        .def_property_readonly("shape", [](const CountMatrix &m)
                               { return py::make_tuple(m.n_gene(), m.n_sample()); })
        .def("__getitem__", get_item)
        .def("__setitem__", set_item)
        .def("__len__", [](const CountMatrix &m)
             { return m.n_gene() * m.n_sample(); })
        .def("__repr__", repr)
        .def("__str__", to_string)
        .def_static("from_file", from_file, py::arg("filepath"), from_file_doc);
}
