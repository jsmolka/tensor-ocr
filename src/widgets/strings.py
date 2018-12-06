import locale

res_load = 1
res_convert = 2
res_save = 3
res_use_dict = 4


resource_strings_de = {
    res_load: "Laden",
    res_convert: "Konvertieren",
    res_save: "Speichern",
    res_use_dict: "Wörterbuch benutzen"
}

resource_strings_en = {
    res_load: "Load",
    res_convert: "Convert",
    res_save: "Save",
    res_use_dict: "Use dictionary"
}


def tr(resource_str):
    """Returns a string in the correct locale."""
    if resource_str not in resource_strings_de:
        return ""

    if "de_DE" in locale.getdefaultlocale():
        return resource_strings_de[resource_str]
    else:
        return resource_strings_en[resource_str]
