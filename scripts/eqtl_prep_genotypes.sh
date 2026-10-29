#!/usr/bin/env bash
# Prepare chr22 genotypes for the cis-eQTL benchmark from the Geuvadis genotype release (E-GEUV-1).
# Filters: the 462 RNA-seq individuals; biallelic SNVs; called in all 462 (no missing GT);
# MAF >= 0.05 in the 462. FORMAT reduced to GT so every tool reads the same hard calls.
set -euo pipefail
W=${EQTL_WORK:-/mnt/pikachu/rapidglm/validation/work/eqtl}
CHR=${1:-22}
IN=$W/downloads/geuvadis/genotypes/GEUVADIS.chr${CHR}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz
OUT=$W/data/genotypes
PLINK2=$W/tools/plink2/plink2
SAMPLES=$W/data/expression/samples.txt
RUN="taskset -c 0-7,12-15 nice -n 10"
mkdir -p $OUT
cd $OUT

# counts before filtering (records, SNVs) for the README
zcat $IN | grep -vc '^#' > chr${CHR}.n_input_records.txt
# The source header lacks ##contig and the FORMAT/PP, FORMAT/BD definitions, which bcftools needs
# in sample-subset mode; declare them (GRCh37 chr lengths) before filtering.
declare -A LEN=([1]=249250621 [2]=243199373 [3]=198022430 [4]=191154276 [5]=180915260 [6]=171115067
  [7]=159138663 [8]=146364022 [9]=141213431 [10]=135534747 [11]=135006516 [12]=133851895
  [13]=115169878 [14]=107349540 [15]=102531392 [16]=90354753 [17]=81195210 [18]=78077248
  [19]=59128983 [20]=63025520 [21]=48129895 [22]=51304566)
zcat $IN | awk -v c=$CHR -v l=${LEN[$CHR]} 'BEGIN{h=1} h && /^#CHROM/{
    print "##contig=<ID=" c ",length=" l ",assembly=GRCh37>";
    print "##FORMAT=<ID=PP,Number=.,Type=Float,Description=\"undeclared in source\">";
    print "##FORMAT=<ID=BD,Number=.,Type=Float,Description=\"undeclared in source\">"; h=0} {print}' \
  | $RUN bcftools view -S $SAMPLES -m2 -M2 -v snps -Ou \
  | $RUN bcftools +fill-tags -Ou -- -t AN,AC,AF,MAF,F_MISSING \
  | $RUN bcftools view -Ou -i 'INFO/F_MISSING=0' \
  | tee >($RUN bcftools view -H - | wc -l > chr${CHR}.n_snv_complete.txt) \
  | $RUN bcftools view -Ou -i 'INFO/MAF>=0.05' \
  | $RUN bcftools annotate -Ou -x QUAL,FILTER,INFO,^FORMAT/GT -I '%CHROM\_%POS\_%REF\_%ALT\_b37' \
  | $RUN bcftools +fill-tags -Oz -o chr${CHR}.vcf.gz -- -t AN,AC,AF,MAF
tabix -f -p vcf chr${CHR}.vcf.gz
bcftools query -l chr${CHR}.vcf.gz > chr${CHR}.vcf_samples.txt
cmp chr${CHR}.vcf_samples.txt $SAMPLES && echo "sample order matches samples.txt"
bcftools view -H chr${CHR}.vcf.gz | wc -l > chr${CHR}.n_final.txt
bcftools query -f '%CHROM:%POS\n' chr${CHR}.vcf.gz | uniq -d | wc -l > chr${CHR}.n_dup_positions.txt

# PLINK 1 bed/bim/fam (A1 = ALT, A2 = REF, kept in VCF order)
$RUN $PLINK2 --vcf chr${CHR}.vcf.gz --make-bed --out chr${CHR} --threads 12 --memory 8000 > /dev/null
# also PLINK2 pgen for tools that prefer it
$RUN $PLINK2 --vcf chr${CHR}.vcf.gz --make-pgen --out chr${CHR} --threads 12 --memory 8000 > /dev/null

# Genotype PCs from LD-pruned SNVs on this chromosome (prototype: chr22 only)
$RUN $PLINK2 --bfile chr${CHR} --indep-pairwise 50 5 0.2 --out chr${CHR}.prune --threads 12 --memory 8000 > /dev/null
$RUN $PLINK2 --bfile chr${CHR} --extract chr${CHR}.prune.prune.in --pca 10 --out chr${CHR}.pca --threads 12 --memory 8000 > /dev/null
wc -l < chr${CHR}.prune.prune.in > chr${CHR}.n_pruned.txt
echo done
