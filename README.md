# molclub
General cheminformatics packages built on top of RDKit.

to-do:
    * ensemble class?
    * finish work on crest
    * vectorize xtb methods to run in same directory. compare performance to see if using restart increases calculation speed.
    * write conf_to_xyz method in utils

package structure?
    * computational stuff
        * mmff
        * gfnff
        * xtb
        * orca
    * methods
        * sp
        * opt
        * freq
    * utils
        * rdkit
        * conf_tools
        * visual