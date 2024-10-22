#!/usr/bin/env nextflow

params.projectDir = "/path/to/working/dir/" 

params.reads = "${params.projectDir}/SRR_downloads/SRR*_{1,2}.fastq"
params.outdir = "${params.projectDir}/results"

params.bowtie2index = "${params.projectDir}/bowtie2_hg38/b2hg38"
params.gtffile = "${params.projectDir}/UCSChg38_human.gtf"

params.fastqc = "${params.outdir}/fastqc"
params.multiqc = "${params.outdir}/multiqc"

params.dnabert2modeldir= "${params.projectDir}/DNABert2_rnaseq"
params.dnabert2csvdir= "${params.projectDir}/DNABert2_rnaseq/csvdir"
params.dnabert2predictdir = "${params.projectDir}/DNABERT2_prediction"
params.bowtie2covindex = "${params.projectDir}/bowtie2_hg38/cov_rbd_bt2"
params.rbdref = "${params.projectDir}/GISAID_ref_genome_RBD_nucleotides.fasta"

log.info """\
    R N A S E Q - N F   P I P E L I N E
    ===================================
    bowtie2 host index      : ${params.bowtie2index}
    bowtie2 cov index       : ${params.bowtie2covindex}
    cov RBD reference       : ${params.rbdref}
    reads                   : ${params.reads}
    outdir                  : ${params.outdir}
    dnabert2 model dir      : ${params.dnabert2modeldir}
    dnabert2 csv dir        : ${params.dnabert2csvdir}
    dnabert2 predict dir    : ${params.dnabert2predictdir}
    gtf file                : ${params.gtffile}
    fastqc dir              : ${params.fastqc}
    multiqc dir             : ${params.multiqc}
    """
    .stripIndent()


// Step 1) Process to trim the fastq files using Trim Galore (paired-end reads)
process TRIM_GALORE {
    
    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("${sample_id}_trimmed_val_1.fq"), path("${sample_id}_trimmed_val_2.fq")

    script:
    """
    trim_galore --paired -q 20 --basename ${sample_id}_trimmed ${reads[0]} ${reads[1]}
    """
}

// Step 2) Process to align the trimmed fastq files using Bowtie2
process BOWTIE2 {
    tag "Bowtie2 on $sample_id"
    
    input:
    tuple val(sample_id), path(reads_1), path(reads_2)

    output:
    tuple val(sample_id), path("${sample_id}_host.sam")

    script:
    """
    # Pass index_base directly inside script & task.cpus placeholder for thread counts can be passed as -process.cpus 8
    bowtie2 -x ${params.bowtie2index} -1 ${reads_1} -2 ${reads_2} -S ${sample_id}_host.sam -p ${task.cpus}
    
    """
}

// Step 3) Process to extract aligned and unaligned reads from alignment sam files 
process SAMTOOLS_aligned {
    tag "SAMtools on $sample_id"
    publishDir params.outdir, mode:'copy'

    input:
    tuple val(sample_id), path(samfile)

    output:
    tuple val(sample_id), path("${sample_id}_aligned_sorted.bam")

    script:
    """
    # Convert SAM to BAM
    samtools view -@ ${task.cpus} -bS ${sample_id}_host.sam > ${sample_id}_host.bam
    
    # Filter aligned reads (using -F 4) and write to BAM
    samtools view -@ ${task.cpus} -b -F 4 ${sample_id}_host.bam > ${sample_id}_aligned.bam
    # Sort and index aligned reads
    samtools sort -@ ${task.cpus} ${sample_id}_aligned.bam -o ${sample_id}_aligned_sorted.bam

    """
}

process SAMTOOLS_unaligned {
    tag "SAMtools on $sample_id"
    publishDir params.outdir, mode:'copy'

    input:
    tuple val(sample_id), path(samfile)

    output:
    tuple val(sample_id), path("${sample_id}_unaligned_1.fastq"), path("${sample_id}_unaligned_2.fastq")

    script:
    """
    # Convert SAM to BAM
    samtools view -@ ${task.cpus} -bS ${sample_id}_host.sam > ${sample_id}_host.bam
    
    # Filter unaligned reads (using -f 4) and write to BAM
    samtools view -@ ${task.cpus} -b -f 4 ${sample_id}_host.bam > ${sample_id}_unaligned.bam
    # Sort and index unaligned reads
    samtools sort -@ ${task.cpus} ${sample_id}_unaligned.bam -o ${sample_id}_unaligned_sorted.bam
    
    # Convert unaligned BAM to FASTQ
    samtools fastq -@ ${task.cpus} -1 ${sample_id}_unaligned_1.fastq -2 ${sample_id}_unaligned_2.fastq -0 /dev/null -s /dev/null ${sample_id}_unaligned_sorted.bam
    
    """
}
    

// Step 4) Process to run StringTie to generate GTF files from host-aligned BAM files
process STRINGTIE {
    publishDir "${params.outdir}", mode: 'copy'  // Move to final destination

    input:
    tuple val(sample_id), path(alignedbam)

    output:
    tuple val(sample_id), path("${sample_id}_aligned.gtf")

    script:
    """
    stringtie ${sample_id}_aligned_sorted.bam -G ${params.gtffile} -p ${task.cpus} -e -o ${sample_id}_aligned.gtf
    
    """
}

// Step 5) QC steps
process FASTQC_pretrim {
    tag "FASTQC before trimgalore on $sample_id"
    publishDir params.fastqc, mode:'copy'

    input:
    tuple val(sample_id), path(reads)

    output:
    path "fastqc_${sample_id}_logs_pre"

    script:
    """
    mkdir fastqc_${sample_id}_logs_pre
    fastqc -o fastqc_${sample_id}_logs_pre -f fastq ${reads[0]} ${reads[1]}
    """
}

process FASTQC_posttrim {
    tag "FASTQC after trimgalore on $sample_id"
    publishDir params.fastqc, mode:'copy'

    input:
    tuple val(sample_id), path(reads_1), path(reads_2)

    output:
    path "fastqc_${sample_id}_logs_post"

    script:
    """
    mkdir fastqc_${sample_id}_logs_post
    fastqc -o fastqc_${sample_id}_logs_post -f fastq ${reads_1} ${reads_2}
    """
}

process MULTIQC {
    publishDir params.multiqc, mode:'copy'

    input:
    path '*' 

    output:
    path 'multiqc_report.html'

    script:
    """
    multiqc .
    """
}

// Step 6) Process to generate the sample list with full paths to GTF files
process GENERATE_GTF_PATHS {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path gtf_files

    output:
    path "sample_list_human.txt"

    script:
    """
    # Create sample list with full paths to GTF files
    > sample_list_human.txt
    for gtf_file in ${gtf_files}; do
        echo "\$(basename "\$gtf_file" .gtf) \$(realpath \$gtf_file)" >> sample_list_human.txt
    done
    """
}

// Step 7) Process to convert GTF files to a count matrix using StringTie’s Python script
process prepDE {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path gtfpath

    output:
    path "host_gene_count_matrix.csv" 

    script:
    """
    python ${params.projectDir}/scripts/prepDE.py -i ${gtfpath} -g host_gene_count_matrix.csv
    """
}

// Step 8) Process to convert GTF files to a count matrix using StringTie’s Python script
process DNABERT2 {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    tuple val(sample_id), path(unalignedfq1), path(unalignedfq2)

    output:
    tuple val(sample_id), path("${sample_id}_prediction.csv")

    script:
    """
    python ${params.projectDir}/scripts/mrna_predictor.py --fastq_path ${sample_id}_unaligned_1.fastq \
                                                          --model_dir ${params.dnabert2modeldir}
                                                          --csv_dir ${params.dnabert2csvdir} \
                                                          --prediction_output_dir ${params.dnabert2predictdir} \
                                                          --virus_threshold 0.95 --variant_threshold 0.95 --batch_size 2048 \
                                                          --rbd_fasta_file ${params.rbdref} \
                                                          --bowtie2_index ${params.bowtie2covindex} \
                                                          --ref_genome ${params.rbdref}
    
    """
}

process circos_generator {
    input:
    tuple val(sample_id), path(predictioncsv)
    
    output:
    path("${sample_id}_circos.jpeg")

    script:
    """
    python ${params.projectDir}/scripts/circos_generator.py --csv ${sample_id}_prediction.csv --out_dir ${params.dnabert2predictdir}
    """
}


workflow {
    // Create channel with the original FASTQ pairs
    read_pairs_ch = Channel.fromFilePairs(params.reads, checkIfExists: true)
    read_pairs_ch_copy = Channel.fromFilePairs(params.reads, checkIfExists: true) // Creating copy channel for fastqc


    // Run TRIM_GALORE to get trimmed FASTQ pairs
    trimmed_pairs_ch = TRIM_GALORE(read_pairs_ch)

    // Pass the trimmed files to BOWTIE2 for alignment
    align_ch = BOWTIE2(trimmed_pairs_ch)
    
    // Pass the sam file to SAMTOOLS to get aligned and unaligned bam file
    sam_aligned_ch = SAMTOOLS_aligned(align_ch)
    sam_unaligned_ch = SAMTOOLS_unaligned(align_ch)

    // Pass the host aligned bam files to STRINGTIE to get gtf file
    gtf_ch = STRINGTIE(sam_aligned_ch)
    
    // Generate FASTQC report for each sample id
    fastqc1_ch = FASTQC_pretrim(read_pairs_ch_copy)
    fastqc2_ch = FASTQC_posttrim(trimmed_pairs_ch)
    MULTIQC(fastqc2_ch.mix(fastqc1_ch).collect()) 
    
    // For each GTF file output by STRINGTIE, append the sample ID and file path to sample_list_human.txt
    gtf_files_ch = gtf_ch.map { it[1] }.collect()  // Map to sample ID and GTF file path
    path_ch = GENERATE_GTF_PATHS(gtf_files_ch)
    
    // Generate count matrix for all gtf files
    matrix_ch = prepDE(path_ch)
    
    // Generate predictions for unaligned reads & a circos plot for distribution
    dnabert2_ch = DNABERT2(sam_unaligned_ch)
    circos_ch = circos_generator(dnabert2_ch)
    
}

