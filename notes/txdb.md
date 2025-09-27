# TxDb (Transcript Database)

儲存 genome metadata，包含：

1. genes
2. transcripts
3. exons (外顯子)
4. CDS (coding sequences)
5. UTR 區域

TxDb 通常會從 GTF/GFF 檔案來建立，RNA-seq 分析會使用到。

## GFF (General Feature Format)

描述 genome 位置、屬性的檔案格式，包含九個欄位：

1. seqname: 染色體名稱 (e.g., chr1)
2. source: 註釋來源
3. feature: 特徵類型 (e.g., gene, exon, CDS)
4. start: 起始位置
5. end: 結束位置
6. score: (optional) 分數
7. strand: 正負
8. frame: 0, 1, 2
9. attribute: 屬性資訊 (e.g., gene_id, transcript_id)

Examples:

```
chr1    HAVANA    gene    11869    14409    .    +    .    gene_id "ENSG00000223972"
chr1    HAVANA    exon    11869    12227    .    +    .    gene_id "ENSG00000223972"
```
