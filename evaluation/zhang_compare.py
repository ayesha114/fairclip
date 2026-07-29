"""Compare FairCLIP results to Zhang et al. CVPR 2025 locked targets."""

# Zhang FairFace targets (the numbers to beat)
ZHANG = {
    "gender": {"MaxSkew": 0.080, "NDKL": 0.025, "ABLE": 78.35},
    "age":    {"MaxSkew": 0.608, "NDKL": 0.294, "ABLE": 60.61},
    "race":   {"MaxSkew": 0.353, "NDKL": 0.125, "ABLE": 69.14},
}

def compare(attribute, maxskew, ndkl, able):
    t = ZHANG[attribute]
    return {
        "attribute": attribute,
        "MaxSkew": maxskew, "MaxSkew_target": t["MaxSkew"],
        "MaxSkew_beats": maxskew < t["MaxSkew"],
        "NDKL": ndkl, "NDKL_target": t["NDKL"], "NDKL_beats": ndkl < t["NDKL"],
        "ABLE": able, "ABLE_target": t["ABLE"], "ABLE_beats": able > t["ABLE"],
    }

if __name__ == "__main__":
    # quick self-test
    print(compare("race", 0.30, 0.10, 70.5))
