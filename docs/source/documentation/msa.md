# MSA

## What is an MSA?

An MSA (multiple sequence alignment) is a file that contains a sequence of amino acids and a number of aligned variant sequences. The aim is to stored information about the evolution of the main sequence. Using this for training allows to reduce the number of model parameters, and use evolution information in the model as explained in the [MSA Transformers papers](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1.full.pdf).

The MSA are generated using [hh-suite](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3019-7) tools, searching similar sequences in the UniClust30 database.

```{note}
The generated MSA must have the .a3m extension to be used in `bio-transformers`
```

Below, you can see an example of an MSA file. The first line is the main sequence. The following sequences are the most probable variants sequences. The `-` character in the msa sequences corresponds to the alignment character.

```bash
>> head msa_example.a3m
```

```bash
>sp|O24396|PURA_WHEAT Adenylosuccinate synthetase, chloroplastic (Fragment) OS=Triticum aestivum OX=4565 PE=1 SV=1
AAAAAGRGRSFSPAAPAPSSVRLPGRQAPAPAAASALAVEADPAADRVSSLSQVSGVLGSQWGDEGKGKLVDVLAPRFDIVARCQGGANAGHTIYNSEGKKFALHLVPSGILHEGTLCVVGNGAVIHVPGFFGEIDGLQSNGVSCDGRILVSDRAHLLFDLHQTVDGLREAELANSFIGTTKRGIGPCYSSKVTRNGLRVCDLRHMDTFGDKLDVLFEDAAARFEGFKYSKGMLKEEVERYKRFAERLEPFIADTVHVLNESIRQKKKILVEGGQATMLDIDFGTYPFVTSSSPSAGGICTGLGIAPRVIGDLIGVVKAYTTRVGSGPFPTELLGEEGDVLRKAGMEFGTTTGRPRRCGWLDIVALKYCCDINGFSSLNLTKLDVLSGLPEIKLGVSYNQMDGEKLQSFPGDLDTLEQVQVNYEVLPGWDSDISSVRSYSELPQAARRYVERIEELAGVPVHYIGVGPGRDALIYK
>UniRef100_A0A0N4UWB0 Adenylosuccinate synthetase n=1 Tax=Enterobius vermicularis TaxID=51028 RepID=A0A0N4UWB0_ENTVE
--------------------------------------------MNDQKRKAPVIVILGAQFGDEGKGKIVDFLIEKekIQLTARCQGGNNAGHTVV-VNGRKSDFHLLPTGIINEDCYNIIGNGVVVNLDALFKEIEHNEIDKLNgWEKRLMISELAHLVTSMHMQADGQQEKSLSSEKIGTTSKGIGPTYSTKCFRNGIRVGELlGDFEAFSAKFRSLAAFYLKQFPGIEVN---VEEELDNYKKHAVCLKRLgiVGDTITYLDEMRAQGKAILVEGANGAMLDIDFGsflytffchsgTYPFVTSSNATVGGAVTGLGIPPTAITEIIGVVKAYETRVGSGPFPTEQQGKIGEDLQSIGHEVGVTTGRKRRCGWLDLFLLKRSSVINGFTALALTKLDILDNFDEIKVATGYR-IDGKSLKAPPSCAADWSRIELEYKTFSGWKDDVSKIRSFNELPENCKTYVKFIEGFVGVPIKWIGVGEDREALIVM
>UniRef100_A0A139AVD1 Adenylosuccinate synthetase n=1 Tax=Gonapodya prolifera (strain JEL478) TaxID=1344416 RepID=A0A139AVD1_GONPJ
-----------------------------------------ATG-------NKAVVVLGAQWGDEGKGKLVDILTQQADLVARCQGGNNAGHTIV-VDGVKFDFHMLPSGLLGaPSTVSLVGSGVVLHLPSFFEEVKKTESKGVSCANRLFVSDRCHLVFDLHQIVDGLKEGELAShkQEIGTTKKGIGPAYSSKASRGGVRVHHLiaPDFAEFESRFRQMAANKKRRYGDFPYD---VDAEVERYRQYRDLIRPYVVDSVTYVHKALQEGKRVLVEGANAVMLDIDFGTFPYVTSSNTTIGGVCTGLGLPPKSIGKVIGVVKAYTTRVGAGPFPTEQLNEVGEHLQTVGAEFGVTTGRKRRCGWLDAAVLRWSHMINGYDSINLTKLDILDGLPTLRIGIAYKHrATGQVYETFPADLHLLEECDVIYEELPGWKESIGGCKSWDALPENARKYVERIEQLVGVNVEYIGVGVSRDSMITK
```

```{caution}
`finetune` method for MSA transormers and `pass_mode=masked` are not available.
```

## How to use msa-transformers?

The library supports `esm_msa1_t12_100M_UR50S` backend which is based on the [MSA Transformers papers](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1.full.pdf).

Instead of passing a list of sequences of the path of a fasta file, you can pass the path to the folder where the .a3m files are stored.

```python
from biotransformers import BioTransformers

msa_folder = "msa_folder"
bio_trans = BioTransformers("esm_msa1_t12_100M_UR50S",num_gpus=1)
msa_embeddings = bio_trans.compute_embeddings(sequences=msa_folder, pool_mode=("cls","mean"), n_seqs_msa=128)
```

As an MSA is composed of multiple sequences, the results have an extra-dimension, which corresponds to all the aligned sequences.

```python
msa_embeddings['cls'].shape
```

If 100 msa files are in the folder, we have the following dimension for the embeddings. We take le `<CLS>` token of each of the 128 sequences in the files.

```python
>> (100, 128, 768)
```

```{caution}
All files in the msa folder must have at least `n_seqs_msa` sequences in each MSA. If you want to remove files with less than `n_seqs_msa` arguments, you can use the `biotransformers.utils.msa_utils.msa_to_remove` function.
```

```python
msa_to_remove("data_msa_sample/",n_seq=128)
```

```python
>>

3/8 have insufficient number of sequences in MSA.
['data/data_msa_sample/seq130_swissprot.a3m',
 'data/data_msa_sample/seq109_swissprot.a3m',
 'data/data_msa_sample/seq124_swissprot.a3m']
```
