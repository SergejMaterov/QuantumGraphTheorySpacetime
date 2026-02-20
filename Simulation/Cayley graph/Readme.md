Default check (Z24 × S3 × Z3, default S3 generator choices): python verify_cayley_aut_group_cli.py --outdir ./verify_out

Custom S3 generator choices using canonical indices (0–5): python verify_cayley_aut_group_cli.py --zN 24 --k3 3 --s1 1 --s2 3 --outdir ./verify_out

Provide explicit permutations for s1/s2: python verify_cayley_aut_group_cli.py --s1_perm "1,0,2" --s2_perm "0,2,1" --outdir ./verify_out

After running, inspect ./verify_out/verification_report.txt. If any non-left colour-preserving automorphisms are found and you passed --save_extra, the file extra_automorphisms.json will contain the mappings.

