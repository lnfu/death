from ._libdeath import CountMatrix

matrix = CountMatrix.from_file("GSE229869_cell.cycle.rnaseq.counts.txt")

print(matrix)
