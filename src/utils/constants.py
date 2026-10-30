DEFAULT_CLASS_NAMES = ["bg", "ABSZ", "TNSZ", "TCSZ"]
DEFAULT_TYPE_CLASS_NAMES = ["ABSZ", "TNSZ", "TCSZ"]
CLASS_TO_ID = {name: i for i, name in enumerate(DEFAULT_CLASS_NAMES)}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
CHBMIT_CLASS_NAMES = ["bg", "seizure"]
TUSZ_TO_INTERNAL = {
    "absz": "ABSZ",
    "tnsz": "TNSZ",
    "tcsz": "TCSZ",
    "fnsz": "FNSZ",
    "gnsz": "GNSZ",
    "cpsz": "CPSZ",
    "spsz": "SPSZ",
    "mysz": "MYSZ",
    "as": "ABSZ",
    "gts": "TNSZ",
    "gtcs": "TCSZ",
    "ABSZ": "ABSZ",
    "TNSZ": "TNSZ",
    "TCSZ": "TCSZ",
    "FNSZ": "FNSZ",
    "GNSZ": "GNSZ",
    "CPSZ": "CPSZ",
    "SPSZ": "SPSZ",
    "MYSZ": "MYSZ",
    "AS": "ABSZ",
    "GTS": "TNSZ",
    "GTCS": "TCSZ",
}

def class_names_to_id(class_names=None):
    names = list(class_names or DEFAULT_CLASS_NAMES)
    if "bg" not in names:
        names = ["bg", *names]
    return {name: i for i, name in enumerate(names)}

def type_class_names(class_names=None):
    names = list(class_names or DEFAULT_TYPE_CLASS_NAMES)
    return [name for name in names if str(name) != "bg"]

def type_class_names_to_id(class_names=None):
    return {name: i for i, name in enumerate(type_class_names(class_names))}
