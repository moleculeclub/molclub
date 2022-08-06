from random import choice
from typing import List

import py3Dmol  # type: ignore
from rdkit import Chem  # type: ignore

py3dmol_colors = [
    "lightgray",
    "hotpink",
    "salmon",
    "orange",
    "yellow",
    "greenyellow",
    "aquamarine",
    "cyan",
    "lightskyblue",
    "violet",
    "magenta",
]


def mol(
    mol: Chem.rdchem.Mol,
    carbon_color: str = "random",
    dark_mode=True,
) -> None:
    """Use py3Dmol to visualize mol in 3D.

    Examples
    --------
    mol = Chem.MolFromSmiles('OCCCO')
    mol = get_low_energy_conformer(mol)
    display_3d_mol(mol)

    Parameters
    ----------
    mol: `rdkit.Chem.rdchem.Mol`
        The input RDKit mol object with an embedded 3D conformer.
    nonpolar_h: `bool`, default = False
        Whether or not to show nonpolar (C-H) hydrogens"""
    if carbon_color == "random":
        carbon_color = choice(py3dmol_colors)
    elif carbon_color in py3dmol_colors:
        pass
    # else:
    #     raise ValueError(f'{carbon_color} is not a valid color')
    # view = py3Dmol.view(
    #     data=mol_block,
    #     width=400,
    #     height=300,
    #     style={
    #         'stick': {
    #             'colorscheme': f'{carbon_color}Carbon',
    #             'radius':0.25
    #         }
    #     }
    # )
    view = py3Dmol.view(width=400, height=300)
    view.removeAllModels()
    mol_block = Chem.rdmolfiles.MolToMolBlock(mol, includeStereo=True)
    view.addModel(mol_block)
    model = view.getModel()
    model.setStyle(
        {
            "stick": {
                "colorscheme": f"{carbon_color}Carbon",
                "radius": 0.25,
            }
        }
    )
    if dark_mode:
        outline_color = carbon_color
        # background_color = '#111111'
        background_color = "black"
    else:
        outline_color = "black"
        background_color = "white"
    view.setViewStyle(
        {"style": "outline", "color": outline_color, "width": 0.04}
    )
    view.setBackgroundColor(background_color)
    view.zoomTo()
    view.show()


def mols(
    mols: List[Chem.Mol],
    dark_mode=True,
) -> None:
    view = py3Dmol.view(width=400, height=300)
    view.removeAllModels()

    for i, mol in enumerate(mols):
        mol_block = Chem.rdmolfiles.MolToMolBlock(mol, includeStereo=True)
        view.addModel(mol_block)
        model = view.getModel()
        colorscheme = f"{py3dmol_colors[i % len(py3dmol_colors)]}Carbon"
        model.setStyle(
            {
                "stick": {
                    "colorscheme": colorscheme,
                    "radius": 0.25,
                }
            }
        )

    if dark_mode:
        outline_color = "white"
        # background_color = '#111111'
        background_color = "black"
    else:
        outline_color = "black"
        background_color = "white"
    view.setViewStyle(
        {"style": "outline", "color": outline_color, "width": 0.04}
    )
    view.setBackgroundColor(background_color)
    view.zoomTo()
    view.show()
