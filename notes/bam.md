# The BAM Format

BGZF: block compression

Block1 + Block2 + Block3 + ... + EOF_Block

## gzip

+---+---+---+---+---+---+---+---+---+---+
|ID1|ID2|CM |FLG|     MTIME     |XFL|OS | (more-->)
+---+---+---+---+---+---+---+---+---+---+

(if FLG.FEXTRA set)

+---+---+=================================+
| XLEN  |...XLEN bytes of "extra field"...| (more-->)
+---+---+=================================+

+---+---+---+---+==================================+
|SI1|SI2|  LEN  |... LEN bytes of subfield data ...|
+---+---+---+---+==================================+
