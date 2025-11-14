====================================================================
DEATH: Differential Expression Analysis Tool for High-throughput NGS
====================================================================

Development
===========

.. code-block:: bash

   uv pip install -e . --force-reinstall --no-cache-dir
   uv run pybind11-stubgen death._core -o .

References
==========

1. Robinson, M.D., McCarthy, D.J., & Smyth, G.K. (2010). edgeR: a Bioconductor 
   package for differential expression analysis of digital gene expression 
   data. *Bioinformatics*, 26(1), 139-140.
2. Anders, S. & Huber, W. (2010). Differential expression analysis for sequence 
   count data. *Genome Biology*, 11(10), R106.
3. Love, M.I., Huber, W., & Anders, S. (2014). Moderated estimation of fold 
   change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 
   15(12), 550.
4. https://samtools.github.io/hts-specs/SAMv1.pdf
5. https://www.ietf.org/rfc/rfc1952.txt
