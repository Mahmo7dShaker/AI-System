import re

def suggest_mapping(columns: list[str]) -> dict:
    cols = columns

    def find_first(patterns):
        for c in cols:
            name = c.lower().strip()
            for pat in patterns:
                if re.search(pat, name):
                    return c
        return None

    # suggestions
    datetime_col = find_first([r"date", r"time", r"datetime", r"timestamp"])
    plant_id_col = find_first([r"plant", r"site", r"station", r"farm", r"id"])
    target_col = find_first([r"energy", r"kwh", r"power", r"kw", r"output", r"generation"])

    # weather optional
    irr = find_first([r"irradi", r"ghi", r"dni", r"poa"])
    amb_temp = find_first([r"ambient", r"air.*temp", r"temp_air"])
    mod_temp = find_first([r"module", r"panel.*temp", r"cell.*temp"])
    wind = find_first([r"wind"])
    cloud = find_first([r"cloud"])

    return {
        "datetime": datetime_col,
        "plant_id": plant_id_col,
        "target": target_col,
        "weather": {
            "irradiance": irr,
            "ambient_temp": amb_temp,
            "module_temp": mod_temp,
            "wind_speed": wind,
            "cloud_cover": cloud,
        }
    }