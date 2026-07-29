"""
gene_environment.report

Word/CSV/figure report generation for the gene-environment pipeline
(Table 1, Table 2, Table 2b, supplementary gene-annotation tables), plus
the shared utilities they're all built on:

    exposure_labels   Italian -> English exposure label translation
    word_utils        shared python-docx formatting helpers
    db_utils          shared stored-routine call + chromosome/slug helpers
    vcf_utils         shared VCF-header sample-id extraction (bcftools)
    runner            run_all_reports(): generate every report in one call

Also includes a separate exploratory analysis chain, not part of
run_all_reports(): c9_check.py (restricted genotype/environment/C9orf72
merge, writing intermediate CSV/JSON files) followed by
generate_c9_stats.py (per-exposure close-vs-far stratified statistics
and C9ORF72 reporting, consuming c9_check's output).
"""
