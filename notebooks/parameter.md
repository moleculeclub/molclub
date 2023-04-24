# Parameterization Scheme

Force fields contain these terms, in order of easiest to hardest:
* X-H bonds (usually constrained)
* X-X bonds
* X-X-X angles
* X-X-X-X dihedrals
* X --- X non-bonded force

An initial parameterization scheme:
1. pick the elements that will be covered in the scheme
    * for now: H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I
    * we will not attempt to cover organometallic, only biologically/ochem relevant elements
2. for each non-H element, calculate the X-H bond length
    * ie. H2B-H, H3C-H, H2N-H, HO-H, etc.
    * 12 total
3. for each pair of non-H elements, calculate the X-X bond length + PES
    * ie. H2B-BH2, H2B-CH3, H2B-NH2, H2B-OH, etc.
    * 144 total
4. for each triplet of non-H elements, calculate the X-X-X bond angle + PES
    * ie. H3C-CH2-CH3, H2N-CH2-CH3, HO-CH2-CH3
    * 1728 total

Then parameterize basic substructures:
* for amino acids: acetic acid, sodium acetate, methylamine, methylammonium chloride, pyrrolidine, pyrrolidinium chloride, methyl guanidine, methyl imidazole, methyl imidazolium, alcohol, acetamide, methylthiol, 
dimethylsulfide, benzene, phenol, indole, methane, ethane, propane, isobutane, neopentane
* common fgs + all methylations + protonations: ether, ester, amine, ammonium chloride, amidine, amidinium chloride, amide, anisole, alkyl/aryl fluoride/chloride/bromide/iodide, sulfoxide, sulfone, sulfonamide, nitrile, alkyl di/trifluoride, urea, cyclic amide, cyclic urea, imide, imine, hydrazine, hydrazide, alkene, phosphate, phosphite, phosphine, carbonate, carbamate, 2-aminopyridine, 2-amino-pyrazine
* ring systems: cyclopropane, aziridine, epoxide, diazirine, oxaziridine, cyclobutane, azetidine, oxetane, cyclopentane, tetrahydrofuran, dihydrofuran, furan, thiophene, pyrrole, pyrazole, oxazole, isoxazole, thiazole, cyclohexane, piperidine, piperazine, pyridine, pyridone, pyridazine, pyrimidine, pyrazine, triazine
* perform bond/angle DFT scans, start with initial parameters, then fit bond/angle parameters to geometries
* freeze all internal coordinates and parameterize self and cross interactions

Then we move onto the ~70 molecules that have measured solution-state properties:
1. break down the molecule into atom_types
2. find and calculate missing atom_type - atom_type bond parameters and atom_type - atom_type - atom_type angle parameters
    * perform bond/angle DFT scanes, then fit bond/angle parameters to geometries
3. find low energy geometry with crest
4. perform high-temperature xtb MD with fixed dihedral angles. collect snapshots, get DFT energy, and refine bond/angle parameters
5. assign non-bonded parameters from closest basic substructure
6. do dihedral scans to fit dihedral parameters
7. do high-tempurature xtb MD with free dihedral angles. collect snapshots, get DFT energy, and refine all parameters

To parameterize an unseen molecule without adding new parameters:
* For each atom in the molecule, find all possible atom_type matches
    * ie. for a benzyl carbon, find benzyl, alkyl, methyl, etc.
* Then for each bond/angle/dihedral parameter, find the closest available atom_type set
    * sometimes, a well-matched atom_type may be available but maybe there is a dihedral in the molecule which has not been seen before. for those cases, pick the best-matching existing parameter