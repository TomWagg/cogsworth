import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


__all__ = ["kstar_translator", "evol_type_translator", "translate_COSMIC_tables"]

fs = 24

kstar_translator = [
    {'long': 'Main Sequence (Low mass)', 'short': 'MS < 0.7', 'colour': (0.996078, 0.843476, 0.469158, 1.0)},
    {'long': 'Main Sequence', 'short': 'MS', 'colour': (0.996078, 0.843476, 0.469158, 1.0)},
    {'long': 'Hertzsprung Gap', 'short': 'HG', 'colour': (0.939608, 0.471373, 0.094902, 1.0)},
    {'long': 'First Giant Branch', 'short': 'FGB', 'colour': (0.716186, 0.833203, 0.916155, 1.0)},
    {'long': 'Core Helium Burning', 'short': 'CHeB', 'colour': (0.29098, 0.59451, 0.78902, 1.0)},
    {'long': 'Early Asymptotic Giant Branch', 'short': 'EAGB', 'colour': (0.294902, 0.690196, 0.384314, 1.0)},
    {'long': 'Thermally Pulsing Asymptotic Giant Branch', 'short': 'TPAGB',
     'colour': (0.723122, 0.889612, 0.697178, 1.0)},
    {'long': 'Helium Main Sequence', 'short': 'HeMS', 'colour': (0.254627, 0.013882, 0.615419, 1.0)},
    {'long': 'Helium Hertsprung Gap', 'short': 'HeHG', 'colour': (0.562738, 0.051545, 0.641509, 1.0)},
    {'long': 'Helium Giant Branch', 'short': 'HeGB', 'colour': (0.798216, 0.280197, 0.469538, 1.0)},
    {'long': 'Helium White Dwarf', 'short': 'HeWD', 'colour': (0.368166, 0.232828, 0.148275, 1.0)},
    {'long': 'Carbon/Oxygen White Dwarf', 'short': 'COWD', 'colour': (0.620069, 0.392132, 0.249725, 1.0)},
    {'long': 'Oxygen/Neon White Dwarf', 'short': 'ONeWD', 'colour': (0.867128, 0.548372, 0.349225, 1.0)},
    {'long': 'Neutron Star', 'short': 'NS', 'colour': (0.501961, 0.501961, 0.501961, 1.0)},
    {'long': 'Black Hole', 'short': 'BH', 'colour': (0.0, 0.0, 0.0, 1.0)},
    {'long': 'Massless Remnant', 'short': 'MR', 'colour': (1.0, 1.0, 0.0, 1.0)},
    {'long': 'Chemically Homogeneous', 'short': 'CHE', 'colour': (0.647059, 0.164706, 0.164706, 1.0)}
]

# where the colours come from
# for i in [1, 2]:
#     kstar_translator[i]["colour"] = plt.get_cmap("YlOrBr")(0.3 * i)

# for i in [3, 4]:
#     kstar_translator[i]["colour"] = plt.get_cmap("Blues")(0.3 * (i - 2))

# for i in [5, 6]:
#     kstar_translator[i]["colour"] = plt.get_cmap("Greens")(0.3 * (3 - (i - 4)))

# for i in [7, 8, 9]:
#     kstar_translator[i]["colour"] = plt.get_cmap("plasma")(0.1 + 0.2 * (i - 7))

# for i in [10, 11, 12]:
#     kstar_translator[i]["colour"] = plt.get_cmap("copper")(0.1 + 0.2 * (i - 9))

evol_type_translator = [
    None,
    {"sentence": "Initial state", "short": "Init", "long": "Initial state"},
    {"sentence": "a star changed stellar type", "short": "Kstar change", "long": "Stellar type changed"},
    {"sentence": "Roche lobe overflow started", "short": "RLOF start", "long": "Roche lobe overflow started"},
    {"sentence": "Roche lobe overflow ended", "short": "RLOF end", "long": "Roche lobe overflow ended"},
    {"sentence": "the binary entered a contact phase", "short": "Contact", "long": "Binary entered contact phase"},
    {"sentence": "the binary coalesced", "short": "Coalescence", "long": "Binary coalesced"},
    {"sentence": "a common envelope phase started", "short": "CE start", "long": "Common-envelope started"},
    {"sentence": "the common envelope phase ended", "short": "CE end", "long": "Common-envelope ended"},
    {"sentence": "no remnant leftover", "short": "No remnant", "long": "No remnant"},
    {"sentence": "the maximum evolution time was reached", "short": "Max evol time", "long": "Maximum evolution time reached"},
    {"sentence": "the binary was disrupted", "short": "Disruption", "long": "Binary disrupted"},
    {"sentence": "a symbiotic phase started", "short": "Begin symbiotic phase", "long": "Begin symbiotic phase"},
    {"sentence": "a symbiotic phase ended", "short": "End symbiotic phase", "long": "End symbiotic phase"},
    {"sentence": "Blue straggler", "short": "Blue straggler", "long": "Blue straggler"},
    {"sentence": "the primary went supernova", "short": "SN1", "long": "Supernova of primary"},
    {"sentence": "the secondary went supernova", "short": "SN2", "long": "Supernova of secondary"},
]


def translate_COSMIC_tables(tab, kstars=True, evol_type=True, label_type="short", replace_columns=True):
    """Translate COSMIC BSE tables to human-readable labels

    Parameters
    ----------
    tab : :class:`~pandas.DataFrame`
        Evolution table from COSMIC
    kstars : `bool`, optional
        Whether to translate kstar_1 and kstar_2, by default True
    evol_type : `bool`, optional
        Whether to translate evol_type, by default True
    label_type : `str`, optional
        Type of label (either "short" or "long"), by default "short"
    replace_columns : `bool`, optional
        Whether to replace original columns (if not new ones are appended), by default True

    Returns
    -------
    translated_tab : :class:`~pandas.DataFrame`
        The translated table
    """
    if kstars:
        unique_kstars = np.unique(tab[["kstar_1", "kstar_2"]].values).astype(int)
        kstar_1_str = np.array([None for _ in range(len(tab))])
        kstar_2_str = np.array([None for _ in range(len(tab))])
        for kstar in unique_kstars:
            kstar_1_str[tab["kstar_1"] == kstar] = kstar_translator[kstar][label_type]
            kstar_2_str[tab["kstar_2"] == kstar] = kstar_translator[kstar][label_type]

        if replace_columns:
            tab.loc[:, "kstar_1"] = kstar_1_str
            tab.loc[:, "kstar_2"] = kstar_2_str
        else:
            tab.loc[:, "kstar_1_str"] = kstar_1_str
            tab.loc[:, "kstar_2_str"] = kstar_2_str

    if evol_type:
        unique_evol_types = np.unique(tab["evol_type"].values).astype(int)
        evol_type_str = np.array([None for _ in range(len(tab))])
        for evol_type in unique_evol_types:
            evol_type_str[tab["evol_type"] == evol_type] = evol_type_translator[evol_type][label_type]

        if replace_columns:
            tab.loc[:, "evol_type"] = evol_type_str
        else:
            tab.loc[:, "evol_type_str"] = evol_type_str

    return tab
